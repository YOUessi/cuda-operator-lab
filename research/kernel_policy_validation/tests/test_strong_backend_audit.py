"""CPU contracts for R06. Must fail before strong_backend_audit.py exists."""
import importlib
import math
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module():
    return importlib.import_module('strong_backend_audit')


@pytest.mark.parametrize('shape', [(1,1536,8960),(16,64,64),(17,31,63)])
def test_positive_shapes_preserved_even_if_custom_unsupported(shape):
    assert module().validate_shape(shape) == shape


@pytest.mark.parametrize('shape', [(0,16,64),(16,0,64),(-1,32,64),(1,2),(True,16,64),(1,2.5,3)])
def test_invalid_shape_rejected(shape):
    with pytest.raises(ValueError): module().validate_shape(shape)


@pytest.mark.parametrize('shape,expected', [((1,1536,8960),False),((16,1536,8960),True),((17,31,63),False)])
def test_custom_eligibility_does_not_silently_relabel(shape,expected):
    assert module().custom_supported(shape) is expected


def rows(custom=80, compiled=100):
    return [dict(case='ffn_16x64x64',mode='eager',arm=name,median_ns=lat)
            for name,lat in [('inductor',compiled),('cublas_v4',110),('custom_v7',custom),('torch_packed',150)]]


def test_policy_is_fit_from_multiple_calibration_runs():
    assert module().fit_table([rows(),rows()]) == {'ffn_16x64x64|eager':'custom_v7'}


def test_uncertain_small_gain_defaults_to_compiler():
    assert module().fit_table([rows(99),rows(97)]) == {'ffn_16x64x64|eager':'inductor'}


def test_one_run_regression_rejects_switch():
    assert module().fit_table([rows(80),rows(110)]) == {'ffn_16x64x64|eager':'inductor'}


def test_fit_rejects_one_run_duplicate_or_missing_candidates():
    for data in [[rows()], [rows()+[rows()[0]],rows()], [rows(),rows()[:-1]]]:
        with pytest.raises(ValueError): module().fit_table(data)


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_bad_latency_never_used(value):
    bad=rows();bad[0]['median_ns']=value
    with pytest.raises(ValueError):module().fit_table([rows(),bad])


def test_observed_headroom_is_not_accuracy():
    assert module().regret({'a':100,'b':80},'a') == pytest.approx(0.25)
    assert module().regret({'a':100,'b':80},'b') == 0


def test_same_precision_contract_is_explicit():
    c=module().precision_contract()
    assert c['input']=='bfloat16' and c['weights']=='bfloat16'
    assert c['projection_output']=='float32'
    assert c['activation_boundary']=='bfloat16'
    assert c['down_projection_output']=='float32'
    assert c['output']=='float32'
    assert c['reference']=='float64_with_bfloat16_boundaries'


def test_unsupported_v7_falls_back_in_static_policy():
    assert module().static_choice((1,1536,8960),'eager')=='cublas_v4'
    assert module().static_choice((16,64,64),'eager')=='custom_v7'


def test_no_production_edit_or_suppression_in_runner_source():
    text=(ROOT/'strong_backend_audit.py').read_text()
    assert 'suppress_errors = True' not in text
    assert 'fullgraph=True' in text
    assert 'max-autotune-no-cudagraphs' in text
    assert 'allow_bf16_reduced_precision_reduction = False' in text
    assert 'allow_tf32 = False' in text
