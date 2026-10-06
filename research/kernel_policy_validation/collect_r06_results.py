#!/usr/bin/env python3
"""CPU-only R06 evidence verification/collection; no new measurements or fitting to tests."""
from collections import Counter
from pathlib import Path
import argparse
import base64
import csv
import gzip
import hashlib
import itertools
import json
import lzma
import math
import statistics
import audit
import strong_backend_audit as core


def summary_key(row):return row['case'],row['mode'],row['arm']


def validate_group(g):
    labels=g['labels'] if 'labels' in g else list(g['event_ns'])
    if not labels or len(set(labels))!=len(labels):raise ValueError('unique candidates required')
    if set(g['event_ns'])!=set(labels) or set(g['wall_ns'])!=set(labels):raise ValueError('missing arm')
    rounds=len(g['orders']);total=0
    if rounds<2:raise ValueError('at least two paired rounds')
    for label in labels:
        if len(g['event_ns'][label])!=rounds or len(g['wall_ns'][label])!=rounds:raise ValueError('missing round')
    for ridx,orders in enumerate(g['orders']):
        size=len(orders)
        if not size:raise ValueError('empty round')
        for order in orders:
            if len(order)!=len(labels) or set(order)!=set(labels):raise ValueError('bad order')
        for position in range(len(labels)):
            counts=Counter(order[position] for order in orders)
            if len(set(counts.values()))!=1 or set(counts)!=set(labels):raise ValueError('position imbalance')
        for a,b in itertools.combinations(labels,2):
            if sum(order.index(a)<order.index(b) for order in orders)*2!=size:raise ValueError('pair imbalance')
        for label in labels:
            for field in ['event_ns','wall_ns']:
                vals=g[field][label][ridx]
                if len(vals)!=size or any(not math.isfinite(v) or v<=0 for v in vals):raise ValueError('bad samples')
            total+=size
    return total


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    published=root/'collected';published.mkdir(exist_ok=False)
    allrows=[];payload={};intervals=0;errors=[];policyrows=[]
    policy=json.loads((root/'frozen_policy.json').read_text())
    calibration=[];sources={}
    for name in ['cal01','cal02','test01','test02']:
        d=root/name;meta=json.loads((d/'metadata.json').read_text());summary=json.loads((d/'summary.json').read_text())
        groups=json.loads(gzip.decompress((d/'raw.json.gz').read_bytes()))['groups']
        if meta['status']!='complete' or len(groups)!=12 or len(summary)!=46:raise ValueError('incomplete matrix')
        if name.startswith('test') and meta['policy_sha256']!=core.sha(root/'frozen_policy.json'):raise ValueError('changed policy')
        if name.startswith('cal'):calibration.append(summary)
        bykey={summary_key(r):r for r in summary}
        if len(bykey)!=len(summary):raise ValueError('duplicate summary')
        for g in groups:
            intervals+=validate_group(g)
            for label,rounds in g['event_ns'].items():
                row=bykey[(g['case'],g['mode'],label)]
                med=statistics.median([statistics.median(v) for v in rounds])
                p95=audit.quantile([v for r in rounds for v in r],0.95)
                wall=statistics.median([statistics.median(v) for v in g['wall_ns'][label]])
                if (med,p95,wall)!=(row['median_ns'],row['p95_ns'],row['wall_median_ns']):raise ValueError('summary mismatch')
                errors.append(g['errors'][label])
            for a,record in g['pairs_vs_inductor'].items():
                if audit.paired_summary(g['event_ns']['inductor'],g['event_ns'][a])!=record:raise ValueError('paired mismatch')
        for case in meta['cases']:
            before=case['compile_counters_before'];after=case['compile_counters_after']
            if after['stats']['unique_graphs']-before.get('stats',{}).get('unique_graphs',0)!=1:raise ValueError('not one compiled graph')
            if sum(after.get('graph_break',{}).values())-sum(before.get('graph_break',{}).values()):raise ValueError('graph break')
            if case['immutable_check']!='pass':raise ValueError('mutated inputs')
        prows=json.loads((d/'policy_results.json').read_text())
        for r in prows:
            subset={s['arm']:s['median_ns'] for s in summary if s['case']==r['case'] and s['mode']==r['mode']}
            if core.regret(subset,r['choice'])!=r['regret']:raise ValueError('policy regret mismatch')
        allrows.extend(dict(run=name,**r) for r in summary)
        if name.startswith('test'):policyrows.extend(dict(run=name,**r) for r in prows)
        payload[name]=dict(metadata=meta,raw=groups,summary=summary,policy_results=prows)
        sources[name]={f:core.sha(d/f) for f in ['metadata.json','raw.json.gz','summary.json','policy_results.json']}
    if core.fit_table(calibration)!=policy['choices']:raise ValueError('frozen policy not reproduced')
    d=root/'managed01';mm=json.loads((d/'metadata.json').read_text());mg=json.loads(gzip.decompress((d/'raw.json.gz').read_bytes()))
    if mm['status']!='complete' or len(mg)!=6:raise ValueError('managed probe incomplete')
    for g in mg:
        intervals+=validate_group(g)
        errors.extend(g['errors'].values())
        for diag in g['diagnostic'].values():
            if core.sha(diag['trace_path'])!=diag['trace_sha256'] or diag['graph_launch_apis']!=3:raise ValueError('unverified graph trace')
    payload['managed01']=dict(metadata=mm,raw=mg)
    payload['frozen_policy']=policy
    write_csv(published/'all_summary.csv',allrows)
    write_csv(published/'heldout_policy_rows.csv',policyrows)
    aggregates=[]
    for mode in ['eager','graph1']:
        for strategy in ['fixed_cublas','fixed_compiler','legacy_static','frozen_table']:
            rows=[r for r in policyrows if r['mode']==mode and r['policy']==strategy]
            aggregates.append(dict(mode=mode,policy=strategy,cases=len(rows),
                mean_regret_pct=100*statistics.mean(r['regret'] for r in rows),
                max_regret_pct=100*max(r['regret'] for r in rows)))
    write_csv(published/'policy_summary.csv',aggregates)
    core.dump(published/'frozen_policy.json',policy)
    packed=lzma.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode())
    (published/'evidence.json.xz').write_bytes(packed)
    event_payload=json.loads(json.dumps(payload))
    for run in ['cal01','cal02','test01','test02','managed01']:
        for group in event_payload[run]['raw']:group.pop('wall_ns')
    packed_events=lzma.compress(json.dumps(event_payload,separators=(',',':')).encode())
    (published/'event_evidence.json.xz.b64').write_bytes(base64.b64encode(packed_events)+b'\n')
    managedrows=[dict(case=g['case'],explicit_us=g['median_ns']['explicit_graph']/1000,
        managed_us=g['median_ns']['managed_max_autotune']/1000,
        explicit_launches=g['diagnostic']['explicit_graph']['graph_launch_apis'],
        managed_launches=g['diagnostic']['managed_max_autotune']['graph_launch_apis']) for g in mg]
    write_csv(published/'managed_summary.csv',managedrows)
    manifest=dict(schema=6,production_library_sha256=payload['cal01']['metadata']['library_sha256'],
        primary_source_commit=payload['cal01']['metadata']['source_commit'],
        primary_groups=48,primary_candidate_rows=len(allrows),managed_groups=6,event_intervals=intervals,
        checked_candidate_outputs=len(errors),max_abs_error=max(e['max_abs'] for e in errors),
        max_relative_l2=max(e['relative_l2'] for e in errors),
        policy_sha256=core.sha(root/'frozen_policy.json'),sources=sources,
        evidence_sha256=hashlib.sha256(packed).hexdigest(),event_evidence_compressed_sha256=hashlib.sha256(packed_events).hexdigest(),
        notes=['primary numerics also include 24 additional compiler-first-call checks',
               'event-only evidence omits raw synchronized wall samples; full evidence preserves them',
               'no new CUDA sanitizer; no independent reviewer; actual publication must be verified separately'])
    manifest['files']={f.name:dict(bytes=f.stat().st_size,sha256=core.sha(f)) for f in published.iterdir() if f.is_file()}
    core.dump(published/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2));print('POLICY',aggregates)

if __name__=='__main__':main()
