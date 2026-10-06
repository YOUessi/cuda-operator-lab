// R05 research helper. Host C++ only; no new or modified device kernels.
// The caller owns tensors, stream and the original operator shared library.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <climits>
#include <new>

namespace {
using U = std::uint64_t;
using V4 = int (*)(const void*, const void*, float*, float*, U,U,U,void*);
using V7 = int (*)(const void*, const void*, const void*, float*, U,U,U,void*);
struct Probe {
  V4 v4=nullptr; V7 v7=nullptr;
  const void* x=nullptr; const void* packed=nullptr;
  float* workspace=nullptr; float* out=nullptr;
  U m=0,k=0,n=0;
  cudaStream_t stream=nullptr;
  cudaEvent_t start=nullptr,end=nullptr;
  cudaGraph_t graphs[2]={nullptr,nullptr};
  cudaGraphExec_t execs[2]={nullptr,nullptr};
  ~Probe() {
    for(int i=0;i<2;++i) {
      if(execs[i]) cudaGraphExecDestroy(execs[i]);
      if(graphs[i]) cudaGraphDestroy(graphs[i]);
    }
    if(start) cudaEventDestroy(start);
    if(end) cudaEventDestroy(end);
  }
  int launch(int arm) {
    if(arm==0) return v4(x,packed,workspace,out,m,k,n,stream);
    const auto* up=static_cast<const unsigned char*>(packed)+2*n*k;
    return v7(x,packed,up,out,m,k,n,stream);
  }
};
bool valid_arm(int arm) { return arm==0 || arm==1; }
}

extern "C" int r05_abi() { return 1; }

extern "C" int r05_create(void* fn4, void* fn7, const void* x,
    const void* packed, float* workspace, float* out,
    U m,U k,U n,void* stream,void** result) {
  if(!result) return cudaErrorInvalidValue;
  *result=nullptr;
  if(!fn4 || !fn7 || !x || !packed || !workspace || !out || !stream ||
     !m || !k || !n || m%16 || k%16 || n%64 ||
     m>INT_MAX || k>INT_MAX || n>INT_MAX/2) return cudaErrorInvalidValue;
  Probe* p=new(std::nothrow) Probe;
  if(!p) return cudaErrorMemoryAllocation;
  p->v4=reinterpret_cast<V4>(fn4); p->v7=reinterpret_cast<V7>(fn7);
  p->x=x; p->packed=packed; p->workspace=workspace; p->out=out;
  p->m=m; p->k=k; p->n=n; p->stream=reinterpret_cast<cudaStream_t>(stream);
  auto status=cudaEventCreate(&p->start);
  if(status==cudaSuccess) status=cudaEventCreate(&p->end);
  if(status!=cudaSuccess) { delete p; return status; }
  *result=p;
  return cudaSuccess;
}

extern "C" int r05_launch(void* handle,int arm) {
  if(!handle || !valid_arm(arm)) return cudaErrorInvalidValue;
  return static_cast<Probe*>(handle)->launch(arm);
}

extern "C" int r05_capture(void* handle,int arm) {
  if(!handle || !valid_arm(arm)) return cudaErrorInvalidValue;
  auto* p=static_cast<Probe*>(handle);
  if(p->graphs[arm] || p->execs[arm]) return cudaErrorInvalidValue;
  auto status=cudaStreamSynchronize(p->stream);
  if(status!=cudaSuccess) return status;
  status=cudaStreamBeginCapture(p->stream,cudaStreamCaptureModeThreadLocal);
  if(status!=cudaSuccess) return status;
  int launch_status=p->launch(arm);
  cudaGraph_t graph=nullptr;
  status=cudaStreamEndCapture(p->stream,&graph);
  if(launch_status!=cudaSuccess || status!=cudaSuccess) {
    if(graph) cudaGraphDestroy(graph);
    return launch_status!=cudaSuccess ? launch_status : static_cast<int>(status);
  }
  p->graphs[arm]=graph;
  return cudaGraphInstantiate(&p->execs[arm],graph,nullptr,nullptr,0);
}

extern "C" int r05_measure(void* handle,int arm,int graph_mode,
                           double* milliseconds,double* wall_ns) {
  if(!handle || !valid_arm(arm) || (graph_mode!=0 && graph_mode!=1) ||
     !milliseconds || !wall_ns) return cudaErrorInvalidValue;
  auto* p=static_cast<Probe*>(handle);
  if(graph_mode && !p->execs[arm]) return cudaErrorInvalidValue;
  const auto wall_start=std::chrono::steady_clock::now();
  auto status=cudaEventRecord(p->start,p->stream);
  if(status!=cudaSuccess) return status;
  int code=graph_mode ? static_cast<int>(cudaGraphLaunch(p->execs[arm],p->stream)) : p->launch(arm);
  if(code!=cudaSuccess) return code;
  status=cudaEventRecord(p->end,p->stream);
  if(status!=cudaSuccess) return status;
  status=cudaEventSynchronize(p->end);
  const auto wall_end=std::chrono::steady_clock::now();
  if(status!=cudaSuccess) return status;
  float elapsed=0;
  status=cudaEventElapsedTime(&elapsed,p->start,p->end);
  if(status!=cudaSuccess) return status;
  *milliseconds=elapsed;
  *wall_ns=std::chrono::duration<double,std::nano>(wall_end-wall_start).count();
  return cudaSuccess;
}

extern "C" int r05_destroy(void* handle) {
  if(!handle) return cudaErrorInvalidValue;
  auto* p=static_cast<Probe*>(handle);
  auto status=cudaStreamSynchronize(p->stream);
  delete p;
  return status;
}
