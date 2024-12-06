import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
import time

def extract_data_from_page(driver: WebDriver, url: str):
    driver.get(url)
    time.sleep(4)
    
    data = {
        "url": url,
        "img_src": None,
        "pre_text": None,
        "p_text": None
    }
    
    try:
        img_element = driver.find_element(By.CSS_SELECTOR, "img[alt='Slide 1']")
        data["img_src"] = img_element.get_attribute("src")
    except Exception as e:
        print(f"Image src not found on {url}: {e}")

    try:
        pre_element = driver.find_element(By.CSS_SELECTOR, "pre")
        data["pre_text"] = pre_element.text
    except Exception as e:
        print(f"Pre tag text not found on {url}: {e}")

    try:
        p_element = driver.find_element(By.CSS_SELECTOR, "div.flex.w-full.items-center.justify-end.lg\\:order-1 > a.flex.items-center > p.text-base.font-semibold.text-dark\\/3")
        data["p_text"] = p_element.text
    except Exception as e:
        print(f"P tag text not found on {url}: {e}")
    
    return data

def main():
    links = [
        "https://modai.fashion/pdp/Ddb9",
        "https://modai.fashion/pdp/DeJV",
        "https://modai.fashion/pdp/DezW",
        "https://modai.fashion/pdp/DebQ",
        "https://modai.fashion/pdp/Deif",
        "https://modai.fashion/pdp/DdDV",
        "https://modai.fashion/pdp/Dewn",
        "https://modai.fashion/pdp/DeYB"
    ]

    driver = webdriver.Chrome()
    driver.implicitly_wait(10)

    data_list = []
    
    try:
        for link in links:
            print(f"Processing {link}...")
            data = extract_data_from_page(driver, link)
            data_list.append(data)
    
    finally:
        driver.quit()

    df = pd.DataFrame(data_list)
    df.to_csv("extracted_data.csv", index=False)
    print("Data extraction complete. Saved to 'extracted_data.csv'.")

if __name__ == "__main__":
    main()
