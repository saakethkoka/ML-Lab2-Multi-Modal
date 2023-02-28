import json
import pickle
import numpy as np
from base64 import b64decode
from io import BytesIO
from PIL import Image
from multiprocessing import Pool
import os


DATA_FOLDER = '/work/users/skoka/Data/WikipediaImageCaptionsProccessed/'
PICKLE_DIRECTORY = "/work/users/skoka/Data/WikipediaImageCaptionsPickles/processed_data/"
TARGET_LANGUAGES = ['en', 'es', 'fr']

# lists all .json files in the data folder
files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
files.sort()

# each json file has multiple json documents so we need to split them before loading
def load_file_into_json(file):
    with open(file) as f:
        json_list = []
        for line in f:
            try:
                json_list.append(json.loads(line))
            except:
                continue
    return json_list

    # with open(file) as f:
    #     json_list = [json.loads(line) for line in f.read().splitlines()]
    # return json_list

def _image_bytes_to_numpy(image_bytes):
    image_decoded = b64decode(image_bytes)
    image = Image.open(BytesIO(image_decoded)).convert("RGB") # Opens image from the decoded bytes
    image = image.resize((250, 250)) # Resizes all images to 250x250 to remain consistent
    image = np.array(image) # Converts image to numpy array
    return image

# Gets the captions for the targeted languages if they exist
def _process_captions(captions):
    processed_captions = {}
    for caption in captions:
        if caption['language'] in TARGET_LANGUAGES:
            # Sometimes the caption is non existent so we need to catch the error
            try:
                processed_captions[caption['language']] = caption['caption_reference_description']
            except:
                continue
    return processed_captions

def process_data(json_list):
    data = []
    for json_doc in json_list:
        image = _image_bytes_to_numpy(json_doc['b64_bytes'])
        captions = _process_captions(json_doc['wit_features'])
        curr_data = {
            'image': image,
            'captions': captions
        }
        data.append(curr_data)
    return data


def process_data_and_save_to_pickle(file):

    pickle_file_to_save = PICKLE_DIRECTORY + file.split('/')[-1].split('.')[0] + '.pickle'
    if os.path.exists(pickle_file_to_save):
        print("File already exists: " + pickle_file_to_save)
        return
    file = DATA_FOLDER + file
    print("Loading file: " + file)
    json_list = load_file_into_json(file)
    print("Processing data...")
    data = process_data(json_list)
    # remove the file extension and add .pickle
    pickle_file_name = file.split('/')[-1].split('.')[0] + '.pickle'
    print("Saving pickle file: " + pickle_file_name)
    with open(pickle_file_to_save, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return



if __name__ == '__main__':
    pool = Pool(50)
    pool.map(process_data_and_save_to_pickle, files)
