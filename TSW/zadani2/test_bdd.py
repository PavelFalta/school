import pytest
from pytest_bdd import scenarios, given, when, then
from cv3 import BankAccount
# Načtení scénářů


scenarios('bank_account.feature')


@pytest.fixture
def account():
    return BankAccount()

@given('nový bankovní účet')
def new_account():
    pass #již realizováno pomocí fixture
    
@when('vložím 200 Kč')
def deposit_money(account):
    account.deposit(200)

@then('zůstatek je 200 Kč')
def check_balance_200(account):
    assert account.get_balance() == 200

@given('bankovní účet s 200 Kč')
def add_200(account):
    account.deposit(200)

@when('vyberu 100 Kč')
def withdraw_100(account):
    account.withdraw(100)

@then('zůstatek je 100 Kč')
def check_balance_100(account):
    assert account.get_balance() == 100