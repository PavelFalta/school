def discount_available(car_data):
    if not car_data["CZ"]:
        return False
    if not car_data["EU"]:
        return False
    return True

def validatestk(car_data):
    if not car_data["STK"]:
        return False
    if not car_data["driver_age"] >= 21:
        return False
    return True

def car_status(car_data):
    return {
        "discount_available": discount_available(car_data),
        "valid_stk": validatestk(car_data)
    }

print(car_status({"STK": True, "driver_age": 21, "CZ": True, "EU": True}))