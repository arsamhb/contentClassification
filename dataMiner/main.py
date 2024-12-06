import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.getters import getAttribute, getText
import time
import math

def scroll_and_collect_links(driver, max_scrolls):
    collected_links = set()
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0

    while scroll_count < max_scrolls:
        elements = driver.find_elements(By.CSS_SELECTOR, "div.relative.row-span-3.flex.flex-col a")
        for element in elements:
            href = element.get_attribute("href")
            if href:
                collected_links.add(href)
        
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(4)  # Wait for new content to load

        # Check if the scroll height has changed
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:  # No more content to load
            break
        last_height = new_height
        scroll_count += 1
    
    return list(collected_links)


def main():
    url = "https://modai.fashion/"
    
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome()
    driver.get(url)
    driver.implicitly_wait(10)
    
    try:
        # Scroll and collect href values
        print("Collecting href values from the page...")
        links = scroll_and_collect_links(driver, max_scrolls=200)

        # Save the links to a CSV file
        df = pd.DataFrame({"links": links})
        df.to_csv("lead_links.csv", index=False)
        print(f"Collected {len(links)} links. Saved to 'lead_links.csv'.")
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()