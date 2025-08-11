# test/test_dbs.py

import os
from util.af_parser import parse_af_file
from semantics.dbs import Dbs

def test_dbs_ranking_on_AF_ex(af_ex_framework):
    # We use a sufficient depth to ensure the ranking is stable.
    dbs = Dbs(af_ex_framework, max_path_length=8)
    calculated_ranking = [sorted(list(group)) for group in dbs.get_ranking()]
    expected_ranking = [
        ['1', '6'], ['7'], ['2', '5'], ['8'], ['3'], ['4']
    ]
    assert calculated_ranking == expected_ranking

def test_dbs_vectors_on_AF_ex(af_ex_framework):
    # We use max_path_length=5 because it's sufficient to distinguish
    # all arguments and keeps the expected data readable.
    dbs = Dbs(af_ex_framework, max_path_length=5)
    calculated_vectors = dbs.get_discussion_vectors()
    expected_vectors = {
        '1': [0, 0, 0, 0, 0],
        '2': [1, 0, 0, 0, 0],
        '3': [2, -1, 0, 0, 0],
        '4': [2, -1, 2, -3, 1],
        '5': [1, 0, 0, 0, 0],
        '6': [0, 0, 0, 0, 0],
        '7': [1, -2, 3, -1, 2],
        '8': [2, -3, 1, -2, 3]
    }
    assert calculated_vectors == expected_vectors