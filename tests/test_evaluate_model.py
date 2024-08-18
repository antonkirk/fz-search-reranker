import pytest
from fz_search_reranker.evaluate_model import calculate_metrics

def test_calculate_metrics():
    retrieved_cuis = ['C0000001', 'C0000002', 'C0000003', 'C0000004', 'C0000005']
    cuis = ['C0000001', 'C0000003', 'C0000005']
    total_relevant = 3
    ks = [1, 3, 5]

    hitatk, precisionatk, recallatk, mrratk = calculate_metrics(retrieved_cuis, cuis, total_relevant, ks)

    assert hitatk == [1, 1, 1]
    assert precisionatk == [1, 2/3, 3/5]
    assert recallatk == [1/3, 2/3, 1]
    assert mrratk == [1, 1, 1]

def test_calculate_metrics_empty_lists():
    retrieved_cuis = []
    cuis = []
    total_relevant = 0
    ks = [1, 3, 5]

    hitatk, precisionatk, recallatk, mrratk = calculate_metrics(retrieved_cuis, cuis, total_relevant, ks)

    assert hitatk == [0, 0, 0]
    assert precisionatk == [0, 0, 0]
    assert recallatk == [0, 0, 0]
    assert mrratk == [0, 0, 0]

def test_calculate_metrics_no_relevant_cuis():
    retrieved_cuis = ['C0000001', 'C0000002', 'C0000003', 'C0000004', 'C0000005']
    cuis = []
    total_relevant = 0
    ks = [1, 3, 5]

    hitatk, precisionatk, recallatk, mrratk = calculate_metrics(retrieved_cuis, cuis, total_relevant, ks)

    assert hitatk == [0, 0, 0]
    assert precisionatk == [0, 0, 0]
    assert recallatk == [0, 0, 0]
    assert mrratk == [0, 0, 0]

def test_calculate_metrics_all_relevant_cuis():
    retrieved_cuis = ['C0000001', 'C0000002', 'C0000003', 'C0000004', 'C0000005']
    cuis = ['C0000001', 'C0000002', 'C0000003', 'C0000004', 'C0000005']
    total_relevant = 5
    ks = [1, 3, 5]

    hitatk, precisionatk, recallatk, mrratk = calculate_metrics(retrieved_cuis, cuis, total_relevant, ks)

    print(mrratk)
    assert hitatk == [1, 1, 1]
    assert precisionatk == [1, 1, 1]
    assert recallatk == [1/5, 3/5, 5/5]
    assert mrratk == [1, 1, 1]