import requests
import os
import time

BASE_URL = "https://analytics.wikimedia.org/published/datasets/one-off/caption_competition/training/joined/"
FILES_TO_DOWNLOAD_FORMAT = "part-{}-48a6f07e-bb86-4735-aac7-883349f41a28-c000.json.gz"

files_to_download = [FILES_TO_DOWNLOAD_FORMAT.format(str(i).zfill(5)) for i in range(324,400)]


STORAGE_DIRECTORY = "/work/users/skoka/Data/WikipediaImageCaptionsProccessed"

print("Starting download...")

# Download all the files, unzip them and store them in the storage directory
for file in files_to_download:
    start = time.time()
    response = requests.get(BASE_URL + file)
    with open(os.path.join(STORAGE_DIRECTORY, file), "wb") as f:
        f.write(response.content)
    # Unzip the file
    os.system("gunzip {}".format(os.path.join(STORAGE_DIRECTORY, file)))
    end = time.time()
    print("Downloaded and unzipped {} in {} seconds".format(file, end - start))
