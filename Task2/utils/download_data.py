import os
import requests
import pandas as pd
import argparse
import concurrent.futures
from tqdm import tqdm

from configs.config import get_config

config = get_config()

def download_image(index, iiifurl):
    global config
    REGION = config["REGION"]
    SIZE = config["SIZE"]
    ROTATION_ANGLE = config["ROTATION_ANGLE"]
    QUALITY = config["QUALITY"]
    IMAGE_FORMAT = config["IMAGE_FORMAT"]
    OUTPUT_PATH = config["OUTPUT_PATH"]

    identifier = '/' + REGION + '/' + SIZE + '/' + ROTATION_ANGLE + '/' + QUALITY + '.' + IMAGE_FORMAT
    combined_url = iiifurl + identifier 
    try:
        response = requests.get(url=combined_url, stream=True, timeout=100)
        if response.status_code == 200:
            output_file = OUTPUT_PATH + '/' + '{:06}'.format(index) + f'.{IMAGE_FORMAT}'
            with open(output_file, 'wb+') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"Error downloading {combined_url}: {e}")
        return False

def download_data(data_path, output_path, size: int = -1):
    global config
    config["DATA_PATH"] = data_path
    config["OUTPUT_PATH"] = output_path
    
    if os.path.exists(config["OUTPUT_PATH"]):
        print(f"Output path {config['OUTPUT_PATH']} already exists")
        return
    os.makedirs(config["OUTPUT_PATH"], exist_ok=True)
    csv_object = pd.read_csv(config["DATA_PATH"] + '/data/objects.csv', low_memory=False)
    csv_images = pd.read_csv(config["DATA_PATH"] + '/data/published_images.csv')
    merge_df = pd.merge(
        csv_object,
        csv_images,
        how="inner",
        left_on="objectid",
        right_on="depictstmsobjectid",
    )
    merge_df = merge_df.query("classification == 'Painting' and isvirtual == 0")
    merge_df = merge_df["iiifurl"].drop_duplicates().dropna()
    images = merge_df.reset_index(drop=True)
    if size == -1:
        size = len(images)
    tasks = [(i, url) for i, url in enumerate(images[:size])]
        
    successful_downloads = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, *task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading images", unit='image'):
            if future.result():
                successful_downloads += 1

    print(f"Successfully download {successful_downloads} images")