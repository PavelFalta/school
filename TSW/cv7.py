from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import dotenv
import os


def test_navigace(driver):
    about_link = driver.find_element(By.PARTIAL_LINK_TEXT, "Browse")
    about_link.click()

    time.sleep(1)

    assert "prohlizeni.html" in driver.current_url

def test_prihlaseni(driver):
    username_input = driver.find_element(By.NAME, "loginName")
    password_input = driver.find_element(By.NAME, "password")

    
    username_input.send_keys(username)
    password_input.send_keys(password)
    
    password_input.send_keys(Keys.RETURN)
    
    time.sleep(1)
    
    assert "logout" in driver.page_source


dotenv.load_dotenv()
driver = webdriver.Chrome()

username = os.getenv("STAG_USERNAME")
password = os.getenv("STAG_PASSWORD")

try:
    
    driver.get("https://portal.ujep.cz/")
    
    test_prihlaseni(driver)
    print("Test přihlášení prošel.")
    test_navigace(driver)
    print("Test navigace prošel.")




except Exception as e:
    print("Test selhal:", str(e))
finally:
    driver.quit()