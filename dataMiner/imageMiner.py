import os
import requests
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO

def download_images(csv_path, output_dir, log_file):
    try:
        os.makedirs(output_dir, exist_ok=True)
        log = open(log_file, "w")

        df = pd.read_csv(csv_path)

        if "img_src" not in df.columns or "page_name" not in df.columns:
            log.write("ERROR: The CSV file must contain 'img_src' and 'page_name' columns.\n")
            log.close()
            return

        name_tracker = {}

        for index, row in df.iterrows():
            img_src = row['img_src']
            page_name = row['page_name']

            if not isinstance(page_name, str):
                page_name = str(page_name)

            if page_name not in name_tracker:
                name_tracker[page_name] = 1
            else:
                name_tracker[page_name] += 1

            appendix = name_tracker[page_name]
            file_name = f"store__{page_name}-{appendix}.jpg"
            file_path = os.path.join(output_dir, file_name)

            try:
                response = requests.get(img_src, stream=True)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    rgb_img = img.convert("RGB")  # Convert to RGB for JPG
                    rgb_img.save(file_path, "JPEG")  # Save as JPG

                    log.write(f"SUCCESS: Downloaded {img_src} as {file_name}\n")
                else:
                    log.write(f"ERROR: Failed to download {img_src}. Status code: {response.status_code}\n")
                print(index)
            except Exception as e:
                log.write(f"ERROR: Exception occurred while downloading {img_src}. Exception: {e}\n")

        log.close()
        print("Image download complete. Log written to", log_file)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


csv_path = "./../data/post_data/extracted_data_full.csv"
output_dir = "./../data/images/store"
log_file = "./download_images_log.txt"

# Run the script
download_images(csv_path, output_dir, log_file)
