

def is_valid_age(age):
    return 0 <= age <= 120


assert is_valid_age(-1) == False
assert is_valid_age(0) == True
assert is_valid_age(1) == True
assert is_valid_age(119) == True
assert is_valid_age(120) == True
assert is_valid_age(121) == False