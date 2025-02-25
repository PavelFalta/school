

def is_valid_age(age):
    return 18 <= age <= 65

assert is_valid_age(17) == False
assert is_valid_age(18) == True
assert is_valid_age(65) == True
assert is_valid_age(66) == False