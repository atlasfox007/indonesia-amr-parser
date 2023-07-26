import os
from google.cloud import storage

keyfile_path = 'arboreal-timer-393614-6e37b45ee8d6.json'

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path

def download_folder(bucket_name, folder_path, local_directory_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all objects in the folder
    blobs = list(bucket.list_blobs(prefix=folder_path))

    # Download each file to the local directory
    for blob in blobs:
        # Get the relative path of the file inside the folder
        relative_file_path = blob.name[len(folder_path):]

        # If the blob is a directory, skip downloading
        if relative_file_path.endswith('/'):
            continue

        # Get the local file path for downloading
        local_file_path = os.path.join(local_directory_path, relative_file_path)

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file to the local directory
        blob.download_to_filename(local_file_path)

# Replace with your Google Cloud Storage bucket name, folder path, and local path
bucket_name = "amr-ta2-bucket"
folder_path = "indonesia-silver-dataset/"
local_directory_path = "./folder1"

# Remove trailing slashes from the local_directory_path
local_directory_path = os.path.normpath(local_directory_path)

download_folder(bucket_name, folder_path, local_directory_path)