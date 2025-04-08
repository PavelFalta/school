from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains


driver = webdriver.Chrome()
driver.get("https://jqueryui.com/droppable/")

driver.switch_to.frame(0)

ActionChains(driver).drag_and_drop(source, target).perform()


assert driver.find_element(By.TAG_NAME, "h1").text == "Example Domain"


print("Test prošel.")
driver.quit()