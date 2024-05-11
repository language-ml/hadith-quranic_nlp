from clint.textui import progress
import requests
import zipfile
import os

from dotenv import load_dotenv
load_dotenv()

data_url = os.getenv("URL_DATA_NEED_QURANIC_PACKAGE")
name_file = os.getenv("NAME_FILE_NEED_QURANIC_PACKAGE")
destination_folder = os.path.join(os.getcwd(), os.getenv("DIRECTORY_DATA_NEED_QURANIC_PACKAGE"))


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

download_and_extract_data(data_url, destination_folder)
