import pytest
from calculator import Calculator

@pytest.fixture
def calculator():
    return Calculator()

def def_test_add(calculator):
    assert calculator.add(1, 2) == 3
    assert calculator.add(0, 0) == 0

@pytest.mark.parametrize("a, b, result", [
    pytest.param(1, 2, 3, id="positive integers"),
    pytest.param(-1, -1, -2, id="negative integers"),
    pytest.param(0, 0, 0, id="zero"),
    pytest.param(-1, 1, 0, id="negative and positive integers"),
])
def par_test_add(a, b, result):
    assert def_test_add(a, b) == result