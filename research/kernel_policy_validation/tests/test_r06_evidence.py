import importlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def evidence():return importlib.import_module('collect_r06_results')


def group():
    return dict(labels=['a','b'],event_ns={'a':[[10,12],[11,13]],'b':[[8,10],[9,11]]},
        wall_ns={'a':[[20,22],[21,23]],'b':[[18,20],[19,21]]},
        orders=[[['a','b'],['b','a']],[['b','a'],['a','b']]])


def test_valid_raw_group_counts_every_interval():
    assert evidence().validate_group(group())==8


def test_biased_orders_rejected():
    g=group();g['orders'][0]=[['a','b'],['a','b']]
    with pytest.raises(ValueError):evidence().validate_group(g)


def test_missing_samples_rejected():
    g=group();g['event_ns']['a'][0].pop()
    with pytest.raises(ValueError):evidence().validate_group(g)


@pytest.mark.parametrize('bad',[0,-1,float('nan')])
def test_invalid_event_never_published(bad):
    g=group();g['event_ns']['b'][0][0]=bad
    with pytest.raises(ValueError):evidence().validate_group(g)


def test_normalized_shape_stays_identical():
    assert evidence().summary_key({'case':'x','mode':'graph1','arm':'inductor'})==('x','graph1','inductor')
