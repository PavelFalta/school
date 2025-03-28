import requests
import random
import time
import threading

def attack():
    url = "http://127.0.0.1:8500/formular"
    while True:
        jmeno = f"Recept{random.randint(1, 1000)}"
        ingredience = " ".join([f"Ingredience{random.randint(1, 100)}" for _ in range(5)])
        postup = "Postup jak to udelat"
        data = {
            "jmeno": jmeno,
            "ingredience": ingredience,
            "postup": postup
        }
        print(data)
        requests.post(url, data=str(data))
        time.sleep(1)

attack_thread = threading.Thread(target=attack)
attack_thread.start()