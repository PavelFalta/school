from selenium import webdriver
from selenium.webdriver.common.by import By



driver = webdriver.Chrome()
driver.get("https://example.com")


assert driver.find_element(By.TAG_NAME, "h1").text == "Example Domain"


print("Test prošel.")
driver.quit()