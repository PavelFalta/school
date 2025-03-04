import numpy as np


class Kalkulacka:
    def multiply(self, *args):
        return np.prod(np.array(args))


class BankAccount:
    def __init__(self):
        self.balance = 0.0

    def deposit(self, amount):
        self.balance += amount
    
    def withdraw(self, amount):
        self.balance -= amount
    
    def get_balance(self):
        return self.balance
    
