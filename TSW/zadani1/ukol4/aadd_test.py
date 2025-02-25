import pytest
from calculator import Calculator

@pytest.fixture
def calculator():
    return Calculator()

def test_add(a, b, result):
    assert calculator.add(a, b) == result

@pytest.mark.parametrize("a, b, result", [
    pytest.param(1, 2, 3, id="positive integers"),
    pytest.param(-1, -1, -2, id="negative integers"),
    pytest.param(0, 0, 0, id="zero"),
    pytest.param(-1, 1, 10, id="negative and positive integers"),
])
def test_addd(a, b, result):
    assert test_add(a, b) == result