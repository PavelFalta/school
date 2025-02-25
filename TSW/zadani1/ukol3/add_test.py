import pytest
from calculator import Calculator

@pytest.fixture
def calculator():
    return Calculator()

def test_add(calculator):
    assert calculator.add(1, 2) == 3
    assert calculator.add(0, 0) == 0

def test_subtract(calculator):
    assert calculator.subtract(1, 2) == -1
    assert calculator.subtract(0, 0) == 0

def test_divide(calculator):
    assert calculator.divide(1, 2) == 0.5
    assert calculator.divide(0, 1) == 0

    with pytest.raises(ValueError):
        calculator.divide(1, 0)
