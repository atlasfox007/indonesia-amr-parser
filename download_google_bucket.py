from google.cloud import storage

def download_files_from_bucket(bucket_name, source_blob_names, destination_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for source_blob_name in source_blob_names:
        blob = bucket.blob(source_blob_name)

        # Replace 'destination_folder' with the local directory where you want to save the downloaded files
        destination_path = os.path.join(destination_folder, os.path.basename(source_blob_name))

        # Download the file to the specified destination
        blob.download_to_filename(destination_path)
        print(f'Downloaded {source_blob_name} to {destination_path}')

if __name__ == '__main__':
    # Replace the following variables with your specific bucket and file details
    BUCKET_NAME = 'your_bucket_name'
    SOURCE_BLOB_NAMES = ['file1.txt', 'folder/file2.txt']  # Replace with the list of files you want to download
    DESTINATION_FOLDER = '/path/to/your/local/destination/folder'  # Replace with your local destination folder

    download_files_from_bucket(BUCKET_NAME, SOURCE_BLOB_NAMES, DESTINATION_FOLDER)