from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
import time


driver = webdriver.Chrome()
driver.get("https://jqueryui.com/droppable/")

driver.switch_to.frame(0)

print(driver.page_source)

source = driver.find_element(By.ID, "draggable")
target = driver.find_element(By.ID, "droppable")


print(target.text)


ActionChains(driver).drag_and_drop(source, target).perform()

print(target.text)

assert target.text == "Dropped!", "Test neprošel."

print("Test prošel.")
driver.quit()

driver = webdriver.Chrome()
driver.get("https://infinite-scroll.com/demo/full-page/")

print(len(driver.find_elements(By.TAG_NAME, "article")))

old = len(driver.find_elements(By.TAG_NAME, "article"))

driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

time.sleep(1)

print(len(driver.find_elements(By.TAG_NAME, "article")))

new = len(driver.find_elements(By.TAG_NAME, "article"))

assert new > old, "Test neprošel."
print("Test prošel.")
driver.quit()