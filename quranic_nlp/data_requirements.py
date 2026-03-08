"""
quranic_nlp.data_requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility to download the required data files for quranic_nlp.

Usage::

    from quranic_nlp.data_requirements import download_data
    download_data()

Or via CLI::

    quranic_data

Required environment variables (set in a ``.env`` file):

- ``URL_DATA_NEED_QURANIC_PACKAGE``: URL of the data zip archive.
- ``NAME_FILE_NEED_QURANIC_PACKAGE``: Local filename for the downloaded zip.
- ``DIRECTORY_DATA_NEED_QURANIC_PACKAGE``: Subdirectory name inside the package to extract into.
"""

from clint.textui import progress
import requests
import zipfile
import os
import json

from dotenv import load_dotenv

load_dotenv()


def _get_env_var(name):
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f'Required environment variable {name!r} is not set. '
            'Please create a .env file with the necessary variables. '
            'See .env.example for reference.'
        )
    return value


def download_data():
    """Download and extract the required data files into the package data directory."""
    data_url = _get_env_var('URL_DATA_NEED_QURANIC_PACKAGE')
    name_file = _get_env_var('NAME_FILE_NEED_QURANIC_PACKAGE')
    destination_folder_name = _get_env_var('DIRECTORY_DATA_NEED_QURANIC_PACKAGE')

    package_dir = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(package_dir, destination_folder_name)
    os.makedirs(data_directory, exist_ok=True)

    _download_and_extract(data_url, name_file, data_directory)

    config_path = os.path.join(package_dir, 'config', 'settings.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['data_directory'] = data_directory
    with open(config_path, 'w') as f:
        json.dump(config, f)

    print(f'Data successfully downloaded to: {data_directory}')


def _download_and_extract(data_url, name_file, destination_folder):
    print('Downloading data...')
    response = requests.get(data_url, stream=True)
    response.raise_for_status()

    total_length = int(response.headers.get('content-length', 0))
    with open(name_file, 'wb') as f:
        chunks = response.iter_content(chunk_size=1024)
        for chunk in progress.bar(chunks, expected_size=(total_length // 1024) + 1):
            if chunk:
                f.write(chunk)

    print('Download complete. Extracting...')
    try:
        with zipfile.ZipFile(name_file, 'r') as zf:
            zf.extractall(destination_folder)
    finally:
        os.remove(name_file)
    print('Extraction complete.')


def main():
    download_data()


if __name__ == '__main__':
    main()
