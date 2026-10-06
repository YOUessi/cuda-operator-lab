from pathlib import Path
import importlib.util
import pytest


def subject():
    path=Path(__file__).resolve().parents[1]/'build_submission_native.py'
    assert path.exists(), 'explicit runtime build guard missing'
    spec=importlib.util.spec_from_file_location('build_submission_native',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_matching_runtime_accepted():
    subject().validate_needed(['libcudart.so.12','libstdc++.so.6'],12)


@pytest.mark.parametrize('names',[['libcudart.so.11.0'],[],['libcudart.so.12','libcudart.so.11.0']])
def test_wrong_or_missing_runtime_rejected(names):
    with pytest.raises(ValueError):subject().validate_needed(names,12)


def test_readelf_needed_parser():
    text='0x01 (NEEDED) Shared library: [libcudart.so.12]\n0x01 (NEEDED) Shared library: [libc.so.6]'
    assert subject().needed_names(text)==['libcudart.so.12','libc.so.6']
