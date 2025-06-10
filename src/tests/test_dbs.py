# test/test_dbs.py

import os
from util.af_parser import parse_af_file
from semantics.dbs import Dbs

def test_dbs_ranking_on_AF_ex(af_ex_framework):
    """
    Tests the final Dbs ranking on the AF_ex.af file.
    This test confirms that the sorting and grouping logic is correct.
    """
    # 1. Calculate the ranking
    # We use a sufficient depth to ensure the ranking is stable.
    dbs = Dbs(af_ex_framework, max_path_length=10)

    # 2. Get the actual ranking and format it for comparison
    calculated_ranking = [sorted(list(group)) for group in dbs.get_ranking()]

    # 3. Define the expected result
    expected_ranking = [
        ['1', '6'], ['7'], ['2', '5'], ['8'], ['3'], ['4']
    ]

    # 4. Assert that the result matches the expectation
    assert calculated_ranking == expected_ranking, "Final ranking does not match expected output."


def test_dbs_vectors_on_AF_ex(af_ex_framework):
    """
    Tests the discussion vectors (the 'scoring') on the AF_ex.af file.
    This test confirms the core path-counting logic is correct.
    """
    # 1. Calculate the semantics. We use max_path_length=5 because it's
    # sufficient to distinguish all arguments and keeps the expected data readable.
    dbs = Dbs(af_ex_framework, max_path_length=5)

    # 2. Get the calculated discussion vectors
    calculated_vectors = dbs.get_discussion_vectors()

    # 3. Define the "golden" expected vectors for AF_ex with depth 5
    # These are the precise, correct scores for each argument.
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

    # 4. Assert that the calculated vectors dictionary matches the expected one.
    # pytest does a deep comparison of dictionaries, checking both keys and values.
    assert calculated_vectors == expected_vectors, "Discussion vectors do not match expected scores."