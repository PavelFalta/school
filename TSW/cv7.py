from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import dotenv
import os
from selenium.webdriver.support.ui import Select
from time import perf_counter


def test_navigace(driver):
    about_link = driver.find_element(By.PARTIAL_LINK_TEXT, "Browse")
    about_link.click()
    time.sleep(0.3)

    assert "prohlizeni.html" in driver.current_url

def test_prihlaseni(driver):
    username_input = driver.find_element(By.NAME, "loginName")
    password_input = driver.find_element(By.NAME, "password")

    
    username_input.send_keys(username)
    password_input.send_keys(password)
    
    password_input.send_keys(Keys.RETURN)
    
    time.sleep(0.3)
    
    assert "logout" in driver.page_source


dotenv.load_dotenv()
driver = webdriver.Chrome()

username = os.getenv("STAG_USERNAME")
password = os.getenv("STAG_PASSWORD")


try:
    start = perf_counter()
    driver.get("https://portal.ujep.cz/")
    # driver.set_window_size(500, 800)

    try:
        burger_menu = driver.find_element(By.ID, "mobile_menu_display_btn")
        burger_menu.click()
        print("Burger menu found and clicked.")
        time.sleep(3)  
    except Exception as e:
        print("Burger menu not found or not clickable:")

    end = perf_counter()
    print(f"Načtení stránky trvalo {end - start:.2f} sekund.")

    elements = driver.find_elements(By.XPATH, "//*[@aria-label]")
    print(f"Počet ARIA prvků na stránce: {len(elements)}")
    if elements:
        print(f"------ ARIA prvky na stránce ---")
        for element in elements:
            print(element.get_attribute("aria-label"))
        print("------ KONEC ARIA prvků na stránce ---")
    
    test_prihlaseni(driver)
    print("Test přihlášení prošel.")
    test_navigace(driver)
    print("Test navigace prošel.")

    elements = driver.find_elements(By.XPATH, "//*[@aria-label]")
    print(f"Počet ARIA prvků na stránce: {len(elements)}")
    if elements:
        print(f"------ ARIA prvky na stránce ---")
        for element in elements:
            print(element.get_attribute("aria-label"))
        print("------ KONEC ARIA prvků na stránce ---")

    start = perf_counter()
    driver.get("https://portal.ujep.cz/portal/studium/uchazec/eprihlaska.html")
    end = perf_counter()
    print(f"Načtení stránky trvalo {end - start:.2f} sekund.")

    elements = driver.find_elements(By.XPATH, "//*[@aria-label]")
    print(f"Počet ARIA prvků na stránce: {len(elements)}")

    podat_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Submit an application')]")
    podat_button.click()
    time.sleep(0.3)

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

    print("Výběr úspěsný...")

    time.sleep(0.3)

    #7. Ověřte, že v pravém seznamu oboru nachází obory: Ekonomika a management, Regionální rozvoj a veřejná správa, Sociální politika a sociální práce.

    assert "Economics and Management" in driver.page_source
    assert "Regional Development and Public Administration" in driver.page_source
    assert "Social Policy and Social Work" in driver.page_source

    print("Test úspěšný.")


except Exception as e:
    print("Test selhal:", str(e))
    driver.save_screenshot("error.png")
finally:
    driver.quit()