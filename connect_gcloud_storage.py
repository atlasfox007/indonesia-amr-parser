import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= 'arboreal-timer-393614-6e37b45ee8d6.json'

client = storage.Client("arboreal-timer-393614")


bucket = client.get_bucket("amr-ta2-bucket")