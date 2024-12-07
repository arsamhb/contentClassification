import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.chrome.options import Options
import time

def extract_data_from_page(driver: WebDriver, url: str):
    driver.get(url)
    time.sleep(3)  
    
    data = {
        "url": url,
        "img_src": None,
        "caption": None,
        "page_name": None
    }
    
    try:
        img_element = driver.find_element(By.CSS_SELECTOR, "img[alt='Slide 1']")
        data["img_src"] = img_element.get_attribute("src")
    except Exception as e:
        print(f"Image src not found on {url}: {e}")

    try:
        pre_element = driver.find_element(By.CSS_SELECTOR, "pre")
        data["caption"] = pre_element.text
    except Exception as e:
        print(f"Pre tag text not found on {url}: {e}")

    try:
        p_element = driver.find_element(By.CSS_SELECTOR, "div.flex.w-full.items-center.justify-end.lg\\:order-1 > a.flex.items-center > p.text-base.font-semibold.text-dark\\/3")
        data["page_name"] = p_element.text
    except Exception as e:
        print(f"P tag text not found on {url}: {e}")
    
    return data

def main():
    input_file = "../data/lead_links/cleaned/lead_links_pt2.csv"
    try:
        with open(input_file, 'r') as file:
            links = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return

    chrome_options = Options()
    chrome_options.add_argument("--headless")    
    chrome_options.add_argument("--no-sandbox") 
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(10)

    data_list = []
    
    try:
        for link in links:
            if link:
                print(f"Processing {link}...")
                data = extract_data_from_page(driver, link)
                data_list.append(data)
    
    finally:
        driver.quit()

    df = pd.DataFrame(data_list)
    df.to_csv("../data/post_data/extracted_data_pt2.csv", index=False)
    print("Data extraction complete. Saved to 'extracted_data.csv'.")

if __name__ == "__main__":
    main()
    