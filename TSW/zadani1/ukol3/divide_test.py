from calculator import Calculator
import pytest

@pytest.fixture
def calculator():
    return Calculator()

def test_divide(calculator):
    assert calculator.divide(10, 2) == 5
    with pytest.raises(ValueError):
        calculator.divide(10, 0)
