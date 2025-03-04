from cv3 import BankAccount
import pytest


@pytest.fixture
def acc():
    return BankAccount()

def test_deposit(acc):
    amount = 100
    expected_balance = 100

    acc.deposit(amount)
    outcome = acc.get_balance()

    assert expected_balance == outcome