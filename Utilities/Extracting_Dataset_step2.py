#extracting the zip file.
import zipfile
import os

zip_file = list(uploaded.keys())[0]  
extract_dir = "data"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted to '{extract_dir}/'") 
