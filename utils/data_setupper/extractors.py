import zipfile
import tarfile
import os

def extract_tgz(tgz_path, extract_to=None):
    """
    Extracts a .tgz file to a specified directory.

    Args:
    tgz_path (str): The file path of the .tgz file to be extracted.
    extract_to (str): Optional. The directory where files will be extracted.
                      If not specified, extracts to the directory containing the .tgz file.

    Returns:
    str: The path where the files were extracted.
    """
    if extract_to is None:
        extract_to = os.path.dirname(tgz_path)

    # Ensure the extraction directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open the tgz file and extract it
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    
    print(f"Files have been extracted to: {extract_to}")
    return extract_to

def extract_zip(zip_path, extract_to=None):
    """
    Extracts a .zip file to a specified directory.

    Args:
    zip_path (str): The file path of the .zip file to be extracted.
    extract_to (str): Optional. The directory where files will be extracted.
                      If not specified, extracts to the directory containing the .zip file.

    Returns:
    str: The path where the files were extracted.
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    # Ensure the extraction directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open the zip file and extract it
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Files have been extracted to: {extract_to}")
    return extract_to