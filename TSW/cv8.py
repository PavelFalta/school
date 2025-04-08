from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains


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

print(driver.page_source)

driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

print(driver.page_source)