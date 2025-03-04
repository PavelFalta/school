import numpy as np


class Kalkulacka:
    def multiply(self, *args):
        return np.prod(np.array(args))


class BankAccount:
    def __init__(self):
        ...

    def deposit(self, amount):
        ...
    
    def withdraw(self, amount):
        ...
    
    def get_balance(self):
        ...
    
