from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "module_audit.py"
    assert path.exists(), "R03 full-module implementation is not present yet"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import module_audit
    return module_audit


@pytest.mark.parametrize("shape", [(0, 128, 256), (16, 0, 64), (16, 64, 0),
                                   (17, 64, 64), (16, 65, 64), (16, 64, 65)])
def test_rejects_shapes_not_shared_by_both_candidates(shape):
    with pytest.raises(ValueError):
        load().validate_shape(shape)


def test_accepts_shared_shape_and_fixed_stage_contract():
    mod = load()
    assert mod.validate_shape((32, 128, 256)) == (32, 128, 256)
    assert mod.STAGES == ("layernorm_fp32", "cast_bf16", "gemm_swiglu", "down_projection_fp32", "residual_add")


@pytest.mark.parametrize("shape,expected", [((32,128,256), "v7_custom"),
    ((128,128,2048), "v7_custom"), ((32,256,64), "v4_cublas"),
    ((256,128,256), "v4_cublas")])
def test_static_v11_snapshot_is_not_refit(shape, expected):
    assert load().static_choice(shape) == expected


def row(scope="module", mode="graph1", decision="b_faster", ratio=1.25):
    return dict(case="gemm_32x128x256", scope=scope, mode=mode, a="v4_cublas",
                b="v7_custom", decision=decision, speedup_a_over_b=ratio,
                a_median_ns=10000.0, b_median_ns=8000.0)


def test_policy_requires_agreement_and_falls_back_on_uncertainty():
    mod = load()
    agreed = mod.learn_policy([[row()], [row()]], "module")
    assert agreed["gemm_32x128x256|graph1"] == "v7_custom"
    uncertain = mod.learn_policy([[row()], [row(decision="unresolved")]], "module")
    assert uncertain["gemm_32x128x256|graph1"] == "v4_cublas"


def test_policy_is_frozen_before_evaluation():
    mod = load()
    policy = mod.learn_policy([[row()], [row()]], "module")
    # Held-out evidence reverses ranking; evaluation must NOT refit policy.
    test_row = row(ratio=0.5, decision="a_faster")
    assert mod.observed_regret(test_row, policy["gemm_32x128x256|graph1"]) == pytest.approx(1.0)
    assert policy["gemm_32x128x256|graph1"] == "v7_custom"


def test_policy_does_not_mix_isolated_with_module():
    mod = load()
    runs = [[row(scope="isolated", decision="a_faster"), row()],
            [row(scope="isolated", decision="a_faster"), row()]]
    assert mod.learn_policy(runs, "isolated")["gemm_32x128x256|graph1"] == "v4_cublas"
    assert mod.learn_policy(runs, "module")["gemm_32x128x256|graph1"] == "v7_custom"


def test_missing_or_duplicate_calibration_is_rejected():
    mod = load()
    for runs in ([[row()]], [[row()], []], [[row(), row()], [row()]]):
        with pytest.raises(ValueError):
            mod.learn_policy(runs, "module")


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        load().observed_regret(row(ratio=ratio), "v7_custom")


def test_regret_direction():
    mod = load()
    assert mod.observed_regret(row(ratio=1.25), "v4_cublas") == pytest.approx(0.25)
    assert mod.observed_regret(row(ratio=1.25), "v7_custom") == 0
    with pytest.raises(ValueError):
        mod.observed_regret(row(), "unknown")


def test_gpu_module_computes_every_stage_and_preserves_inputs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for complete module test")
    mod = load()
    from cuda_operator_lab import bindings
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        cases = mod.make_module_cases(torch, bindings, (32, 128, 256), 1701)
        module = cases["module"]
        originals = {key: value.clone() for key, value in module["immutable"].items()}
        for fn in module["variants"].values():
            for t in module["scratch"]:
                t.fill_(float("nan"))
            fn()
            stream.synchronize()
            torch.testing.assert_close(module["outputs"]["v4_cublas"], module["reference"], rtol=1e-3, atol=1e-3)
        for key, original in originals.items():
            torch.testing.assert_close(module["immutable"][key], original, rtol=0, atol=0)
        assert module["pointers"]["h"]["data_ptr"] == cases["isolated"]["pointers"]["h"]["data_ptr"]
