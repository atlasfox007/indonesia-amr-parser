import zipfile

# Replace 'archive.zip' with the name of your .zip file
zip_file = "data.zip"

# Specify the destination directory for extraction
destination_directory = "./"

# Open the .zip archive
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    # Extract all files from the archive to the destination directory
    zip_ref.extractall(destination_directory)