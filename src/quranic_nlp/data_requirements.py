from clint.textui import progress
import requests
import zipfile
import os

from dotenv import load_dotenv
load_dotenv()

data_url = os.getenv("URL_DATA_NEED_QURANIC_PACKAGE")
name_file = os.getenv("NAME_FILE_NEED_QURANIC_PACKAGE")
destination_folder = os.path.join(os.getcwd(), os.getenv("DIRECTORY_DATA_NEED_QURANIC_PACKAGE"))

import os
import json
import requests

def download_data():
    data_directory = os.path.join(os.path.dirname(__file__), destination_folder)    
    os.makedirs(data_directory, exist_ok=True)
    
    # دانلود فایل‌ها
    download_and_extract_data(data_url, data_directory)
    
    # به‌روزرسانی فایل پیکربندی
    config_path = os.path.join(os.path.dirname(__file__), 'config/settings.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['data_directory'] = data_directory
    
    with open(config_path, 'w') as f:
        json.dump(config, f)


def download_and_extract_data(data_url, destination_folder):
    print('requsete started')
    response = requests.get(data_url, stream=True)

    with open(name_file, 'wb') as f:
        total_length = int(response.headers.get('content-length'))
        for chunk in progress.bar(response.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()


    print('requsete ended')
    
    if response.status_code == 200:
        print('extractrion started')
        
        with zipfile.ZipFile(name_file, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        
        os.remove(name_file)
        print('extractrion ended')        
        print("داده‌ها با موفقیت دانلود شد و در مسیر مورد نظر ذخیره شد.")
    else:
        os.remove(name_file)
        print("خطا در دانلود داده‌ها!")

download_data()
