from pathlib import Path
import importlib.util
import sys
import math
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def subject():
    path = ROOT / 'submission_audit.py'
    assert path.exists(), 'R05 implementation missing'
    spec = importlib.util.spec_from_file_location('submission_audit', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('shape', [(0,64,64),(17,64,64),(16,31,64),(16,64,63),
                                  (-16,64,64),(16,64),(True,64,64),(16,2**32,64)])
def test_reject_bad_shape(shape):
    with pytest.raises(ValueError):
        subject().validate_shape(shape)


@pytest.mark.parametrize('shape',[(16,64,64),(32,128,256),(16,1536,8960)])
def test_valid_shape(shape):
    assert subject().validate_shape(shape) == shape


def test_modes_and_representative_dimensions():
    mod=subject()
    assert mod.MODES == ('python_binding','python_raw','native_eager','native_graph')
    assert len(mod.SHAPES)==8
    assert (16,1536,8960) in mod.SHAPES and (128,1536,8960) in mod.SHAPES


@pytest.mark.parametrize('event,wall',[(0,1),(-1,1),(math.nan,1),(1,math.inf)])
def test_bad_timing_rejected(event,wall):
    with pytest.raises(ValueError):
        subject().timing_record(event,wall)


def test_milliseconds_are_not_microseconds():
    assert subject().timing_record(0.005,10000)==(5000,10000)


def test_refuse_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        subject().new_output(tmp_path)


def test_create_new_output(tmp_path):
    path=tmp_path/'fresh'
    subject().new_output(path)
    assert path.is_dir()


def test_order_validation_accepts_balanced():
    orders=[[0,1],[1,0],[1,0],[0,1]]
    assert subject().validate_orders(orders,4)


def test_order_validation_rejects_biased():
    with pytest.raises(ValueError):
        subject().validate_orders([[0,1]]*4,4)


def test_order_validation_rejects_missing_arm():
    with pytest.raises(ValueError):
        subject().validate_orders([[0,0],[1,0]],2)


def test_native_source_is_host_only():
    subject()
    source=ROOT/'submission_native.cu'
    assert source.exists(), 'native source missing'
    content=source.read_text()
    assert '__global__' not in content and '__device__' not in content
    for name in ('r05_create','r05_capture','r05_launch','r05_measure','r05_destroy'):
        assert name in content
