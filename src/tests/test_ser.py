# test/test_ser.py

import os
import math  # Required for math.inf
from util.af_parser import parse_af_file
from semantics.ser import Ser

def test_ser_ranking_on_AF_ex(af_ex_framework):
    """
    Tests the final Ser ranking on the AF_ex.af file.
    This test confirms that the sorting and grouping logic is correct.
    """
    # 1. Calculate the ranking
    ser = Ser(af_ex_framework)

    # 2. Get the actual ranking and format it for comparison
    # The list() and sorted() calls ensure a consistent order for assertion
    calculated_ranking = [sorted(list(group)) for group in ser.get_ranking()]

    # 3. Define the expected result based on the thesis example
    # Rank 1 (index 1), Rank 2 (index 2), Rank 3 (index inf)
    expected_ranking = [
        ['1', '6'],
        ['8'],
        ['2', '3', '4', '5', '7']
    ]

    # 4. Assert that the result matches the expectation
    assert calculated_ranking == expected_ranking, "Final ranking does not match expected output."


def test_ser_indices_on_AF_ex(af_ex_framework):
    """
    Tests the serialisation indices (the 'scoring') on the AF_ex.af file.
    This test confirms the core serialisation index calculation is correct.
    """
    # 1. Calculate the semantics
    ser = Ser(af_ex_framework)

    # 2. Get the calculated serialisation indices
    calculated_indices = ser.get_serialisation_indices()

    # 3. Define the "golden" expected indices for AF_ex
    # These are the precise, correct scores for each argument.
    expected_indices = {
        '1': 1,
        '2': math.inf,
        '3': math.inf,
        '4': math.inf,
        '5': math.inf,
        '6': 1,
        '7': math.inf,
        '8': 2
    }

    # 4. Assert that the calculated dictionary matches the expected one.
    # pytest does a deep comparison of dictionaries.
    assert calculated_indices == expected_indices, "Serialisation indices do not match expected scores."