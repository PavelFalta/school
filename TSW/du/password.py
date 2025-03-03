def len_check(password:str, minlen:int):
    if len(password) < minlen: 
        return False
    return True

def numeric_check(password:str, minfreq:int):
    count = 0

    for char in password:
        if char.isdigit():
            count += 1
        if count >= minfreq:
            return True
    
    return False

def upper_check(password:str, minfreq:int):
    count = 0
    
    for char in password:
        if char.isupper():
            count += 1
        if count >= minfreq:
            return True
    
    return False

def lower_check(password:str, minfreq:int):
    count = 0
    
    for char in password:
        if char.islower():
            count += 1
        if count >= minfreq:
            return True
    
    return False
    

def is_valid_password(password:str) -> bool:
    if all([len_check(password, 8), 
            numeric_check(password, 1), 
            upper_check(password, 1),
            lower_check(password, 1)]):
        return True
    return False