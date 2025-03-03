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
    ...