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

def test_minus_deposit(acc):
    amount = -100
    expected_balance = -100

    acc.deposit(amount)
    outcome = acc.get_balance()

    assert expected_balance == outcome

def test_initial_balance(acc):
    expected_balance = 0

    outcome = acc.get_balance()

    assert expected_balance == outcome

def test_inf(acc):    
    amount = float('inf')
    expected_balance = float('inf')

    acc.deposit(amount)
    outcome = acc.get_balance()

    assert expected_balance == outcome

def test_minus_inf(acc):    
    amount = float('-inf')
    expected_balance = float('-inf')

    acc.deposit(amount)
    outcome = acc.get_balance()

    assert expected_balance == outcome
