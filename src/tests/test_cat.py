# tests/test_cat.py

import pytest
from semantics.cat import Cat

def test_cat_strengths_on_AF_ex(af_ex_framework):
    """
    Tests the calculated strengths on the AF_ex framework.
    This confirms the core iterative calculation is correct by comparing
    its output to the results in the thesis.
    """
    # 1. Initialize the Cat semantics calculator
    cat = Cat(af_ex_framework)

    # 2. Get the calculated strengths
    calculated_strengths = cat.get_strengths()

    # 3. Define the expected strengths from the thesis (Example 2.3.5.1)
    # The mapping is a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8.
    expected_strengths = {
        '1': pytest.approx(1.0),      # Cat(a)
        '2': pytest.approx(0.5),      # Cat(b)
        '3': pytest.approx(0.4),      # Cat(c)
        '4': pytest.approx(0.38, abs=1e-2),  # Cat(d)
        '5': pytest.approx(0.5),      # Cat(e)
        '6': pytest.approx(1.0),      # Cat(f)
        '7': pytest.approx(0.65, abs=1e-2), # Cat(g)
        '8': pytest.approx(0.53, abs=1e-2) # Cat(h)
    }

    # 4. Assert that the calculated strengths match the expected values
    # We compare key by key for clearer error messages if something fails.
    for arg in expected_strengths:
        assert arg in calculated_strengths, f"Argument {arg} missing from results."
        assert calculated_strengths[arg] == expected_strengths[arg], \
               f"Strength for argument {arg} is incorrect."

def test_cat_ranking_on_AF_ex(af_ex_framework):
    """
    Tests the final ranking on the AF_ex framework.
    This test confirms the sorting and grouping logic is correct.
    The expected ranking is taken from the thesis.
    """
    # 1. Calculate the ranking
    cat = Cat(af_ex_framework)

    # 2. Get the actual ranking and format it for consistent comparison
    calculated_ranking = [sorted(list(group)) for group in cat.get_ranking()]

    # 3. Define the expected result based on the thesis:
    # Ranking: a ~ f > g > h > c > b ~ e > d
    expected_ranking = [
        ['1', '6'],  # Rank 1: {a, f}
        ['7'],       # Rank 2: {g}
        ['8'],       # Rank 3: {h}
        ['2', '5'],  # Rank 4: {b, e}
        ['3'],       # Rank 5: {c}
        ['4']        # Rank 6: {d}
    ]

    # 4. Assert that the result matches the expectation
    assert calculated_ranking == expected_ranking, "Final ranking does not match expected output."