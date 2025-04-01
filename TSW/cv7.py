from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import dotenv
import os

dotenv.load_dotenv()
driver = webdriver.Chrome()

username = os.getenv("STAG_USERNAME")
password = os.getenv("STAG_PASSWORD")

try:
    
    driver.get("https://portal.ujep.cz/")
    
    username_input = driver.find_element(By.NAME, "loginName")
    password_input = driver.find_element(By.NAME, "password")

    
    username_input.send_keys(username)
    password_input.send_keys(password)
    
    password_input.send_keys(Keys.RETURN)
    
    time.sleep(1)
    
    if "přihlášen" not in driver.page_source:
        raise Exception("Přihlášení selhalo.")
    print("Test přihlášení prošel.")

    about_link = driver.find_element(By.LINK_TEXT, "Prohlížení")
    about_link.click()
    time.sleep(2)
    assert "prohlizeni.html" in driver.current_url
    print("Test navigace prošel.")
except Exception as e:
    print("Test selhal:", str(e))
finally:
    driver.quit()