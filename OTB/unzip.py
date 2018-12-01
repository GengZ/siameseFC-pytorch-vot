import zipfile
import glob


zip_files = glob.glob('./*.zip')
for path_to_zip_file in zip_files:
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall()
    # zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()