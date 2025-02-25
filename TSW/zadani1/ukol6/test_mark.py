from calculator import Calculator
import pytest

@pytest.fixture
def calculator():
    return Calculator()

@pytest.mark.addition
def test_add(calculator):
    assert calculator.add(2, 3) == 5
    assert calculator.add(-1, 1) == 0

c = 66

print(f"{c:b}")