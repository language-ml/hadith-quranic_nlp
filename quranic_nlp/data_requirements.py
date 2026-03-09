"""
quranic_nlp.data_requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Downloads the required data files for quranic_nlp from GitHub Releases.

Usage::

    from quranic_nlp.data_requirements import download_data
    download_data()

Or via CLI after ``pip install quranic-nlp``::

    quranic_data
"""

from clint.textui import progress
import requests
import zipfile
import os
import json

# GitHub Release asset URL — update this when releasing a new data version
DATA_URL = (
    'https://github.com/language-ml/hadith-quranic_nlp'
    '/releases/download/v1.2.1/data.zip'
)
_DATA_DIR_NAME = 'data'
_ZIP_NAME = 'quranic_data.zip'


def _package_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _data_dir():
    return os.path.join(_package_dir(), _DATA_DIR_NAME)


def is_data_available():
    """Return True if the data directory exists and is non-empty."""
    d = _data_dir()
    return os.path.isdir(d) and bool(os.listdir(d))


def download_data(force=False):
    """
    Download and extract the required data files into the package data directory.

    Parameters
    ----------
    force : bool
        Re-download even if data already exists. Default ``False``.

    Example
    -------
    ::

        from quranic_nlp.data_requirements import download_data
        download_data()
    """
    if not force and is_data_available():
        print(f'Data already present at: {_data_dir()}')
        print('Use download_data(force=True) to re-download.')
        return

    data_dir = _data_dir()
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(_package_dir(), _ZIP_NAME)

    _download_and_extract(DATA_URL, zip_path, data_dir)

    print(f'Data successfully installed to: {data_dir}')


def _download_and_extract(url, zip_path, destination):
    print('Downloading data from GitHub Release...')
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_length = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f:
        for chunk in progress.bar(
            response.iter_content(chunk_size=1024),
            expected_size=(total_length // 1024) + 1,
        ):
            if chunk:
                f.write(chunk)

    print('Download complete. Extracting...')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(destination)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    print('Extraction complete.')


def _update_config(data_dir):
    config_path = os.path.join(_package_dir(), 'config', 'settings.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['data_directory'] = data_dir
    with open(config_path, 'w') as f:
        json.dump(config, f)


def main():
    download_data()


if __name__ == '__main__':
    main()
