"""R06 extra diagnostic: graph use requires an observed graph launch, not its mode name."""
import importlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def probe():return importlib.import_module('managed_backend_probe')


def test_no_launch_is_not_a_graph_success():
    assert probe().graph_launch_count(['aten::mm','cudaLaunchKernel','CudaGraphTree'])==0


def test_count_only_actual_graph_launch_apis():
    assert probe().graph_launch_count(['cudaGraphLaunch','cudaGraphLaunch_ptsz','cuGraphLaunch','cudaGraphInstantiate','cudaLaunchKernel'])==3


@pytest.mark.parametrize('name',['max-autotune','max-autotune-no-cudagraphs'])
def test_fullgraph_no_silent_fallback(name):
    options=probe().compile_options(name)
    assert options==dict(fullgraph=True,dynamic=False,mode=name)


def test_unknown_mode_is_not_accepted():
    with pytest.raises(ValueError):probe().compile_options('default')
