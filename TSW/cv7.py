from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import dotenv
import os
from selenium.webdriver.support.ui import Select


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

    driver.get("https://portal.ujep.cz/portal/studium/uchazec/eprihlaska.html")

    podat_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Submit an application')]")
    podat_button.click()
    time.sleep(1)

    faculty_dropdown = driver.find_element(By.NAME, "fakulta")

    select = Select(faculty_dropdown)

    select.select_by_value("FSE")
    
    form_dropdown = driver.find_element(By.NAME, "forma")
    form_select = Select(form_dropdown)
    form_select.select_by_value("K")
    
    type_dropdown = driver.find_element(By.NAME, "typ")
    type_select = Select(type_dropdown)
    type_select.select_by_value("7")

    location_dropdown = driver.find_element(By.NAME, "misto")
    location_select = Select(location_dropdown)
    location_select.select_by_value("U")

    language_dropdown = driver.find_element(By.NAME, "jazyk")
    language_select = Select(language_dropdown)
    language_select.select_by_value("CZ")

    search_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Search')]")
    search_button.click()

    time.sleep(1)

    


except Exception as e:
    print("Test selhal:", str(e))
finally:
    driver.quit()