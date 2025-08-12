# test/test_ser.py

import os
import math  # Required for math.inf
from util.af_parser import parse_af_file
from semantics.ser import Ser

def test_ser_ranking_on_AF_ex(af_ex_framework):
    """
    Tests the final Ser ranking on the AF_ex.af file.
    """
    ser = Ser(af_ex_framework)

    calculated_ranking = [sorted(list(group)) for group in ser.get_ranking()]

    expected_ranking = [
        ['1', '6'],
        ['8'],
        ['2', '3', '4', '5', '7']
    ]
    assert calculated_ranking == expected_ranking, "Final ranking does not match expected output."


def test_ser_indices_on_AF_ex(af_ex_framework):
    """
    Tests the serialisation indices (the 'scoring') on the AF_ex.af file.
    """
    ser = Ser(af_ex_framework)

    calculated_indices = ser.get_serialisation_indices()

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
    assert calculated_indices == expected_indices, "Serialisation indices do not match expected scores."