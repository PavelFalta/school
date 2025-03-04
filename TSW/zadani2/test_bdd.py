import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from cv3 import BankAccount

# Load scenarios
scenarios('bank_account.feature')

@pytest.fixture
def account():
    return BankAccount()

@given('nový bankovní účet')
def new_account():
    pass  # Already implemented using fixture

@given(parsers.parse('bankovní účet s {amount:d} Kč'))
def add_amount(account, amount):
    account.deposit(amount)

@when(parsers.parse('vložím {amount:d} Kč'))
def deposit_money(account, amount):
    account.deposit(amount)

@when(parsers.parse('vyberu {amount:d} Kč'))
def withdraw_money(account, amount):
    account.withdraw(amount)

@then(parsers.parse('zůstatek je {amount:d} Kč'))
def check_balance(account, amount):
    assert account.get_balance() == amount
