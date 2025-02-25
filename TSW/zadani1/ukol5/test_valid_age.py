from calculator import Calculator
import pytest
import numpy as np

def is_valid_age(age):
    return 18 <= age <= 65

assert is_valid_age(17) == False
assert is_valid_age(18) == True
assert is_valid_age(65) == True
assert is_valid_age(66) == False


@pytest.fixture
def calculator():
    return Calculator()

def test_divide_range(calculator):
    range_a = np.arange(1, 66e100, step=1e99)
    range_b = np.arange(1, 66)
    for a, b in zip(range_a, range_b):
        assert calculator.divide(a, b) == a / b