from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
import time
import os

def test1():
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

def test2():
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

def test3():
    download_dir = os.path.abspath(".") # relativní cestu ignoruje, musíte vytvořit absolutní

    options = webdriver.ChromeOptions()

    options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True # jinak nestáhne soubor, ale otevře ho v prohlížeči
    })

    driver = webdriver.Chrome(options=options)

    url = "https://file-examples.com/index.php/sample-documents-download/sample-pdf-download/"

    driver.get(url)
    href="https://file-examples.com/wp-content/storage/2017/10/file-example_PDF_1MB.pdf"

    time.sleep(2)
    consent_button = driver.find_element(By.CLASS_NAME, "fc-cta-consent")
    consent_button.click()

    button = driver.find_element(By.XPATH, "//a[@href='" + href + "']")
    button.click()

    time.sleep(5)

    assert os.path.exists(os.path.join(download_dir, "file-example_PDF_1MB.pdf")), "Test neprošel."
    print("Test prošel.")

    # os.remove(os.path.join(download_dir, "file-example_PDF_1MB.pdf"))
    driver.quit()


def test4():
    driver = webdriver.Chrome()
    driver.get("https://the-internet.herokuapp.com/upload")

    time.sleep(20)

    upload = driver.find_element(By.ID, "file-upload")

    download_dir = os.path.abspath(".")
    file = os.path.join(download_dir, "file-example_PDF_1MB.pdf")
    upload.send_keys(file)

    submit = driver.find_element(By.ID, "file-submit")
    submit.click()

    # headers 3

    headers = driver.find_elements(By.TAG_NAME, "h3")
    assert "File Uploaded!" in headers[0].text, "Test neprošel."
    print("Test prošel.")

    driver.quit()


# test3()
test4()