import os
import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


DRIVER_PATH = (
    r"C:\Users\Bhoomi Vaity\OneDrive\Desktop\Fantasy XI Project"
    r"\Fantasy-XI\.venv\Lib\site-packages\seleniumbase\drivers"
    r"\chromedriver.exe"
)


def scrape_fbref_table(url):

    if not os.path.exists(DRIVER_PATH):
        raise FileNotFoundError(
            f"ChromeDriver not found:\n{DRIVER_PATH}"
        )


    options = Options()

    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    options.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/150 Safari/537.36"
    )


    service = Service(DRIVER_PATH)


    driver = webdriver.Chrome(
        service=service,
        options=options
    )


    try:

        print("Opening FBRef...")

        driver.get(url)

        time.sleep(5)


        print("Page loaded")

        html = driver.page_source


        print("Extracting tables...")

        tables = pd.read_html(html)


        print(
            f"Found {len(tables)} tables"
        )


        return tables


    finally:

        driver.quit()



if __name__ == "__main__":

    url = (
        "https://fbref.com/en/comps/9/"
        "stats/Premier-League-Stats"
    )


    tables = scrape_fbref_table(url)


    print("\nFirst table preview:")
    print(tables[0].head())


    print("\nColumns:")
    print(tables[0].columns.tolist())