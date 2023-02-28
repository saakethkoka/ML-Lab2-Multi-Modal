import json
import os
import time
import pickle
from multiprocessing import Pool

DATA_FOLDER = '/work/users/skoka/Data/WikipediaImageCaptionsProccessed/'
files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
files.sort()
files = files[:300]

# each json file has multiple json documents so we need to split them before loading
def load_file_into_json(file):
    with open(file) as f:
        json_list = [json.loads(line) for line in f.read().splitlines()]
    return json_list

def process_file(file):
    captions = []
    try:
        json_list = load_file_into_json(DATA_FOLDER + file)
    except:
        return captions
    for json_doc in json_list:
        for caption in json_doc['wit_features']:
            if caption["language"] == "en":
                try:
                    english_caption = caption["caption_reference_description"]
                    captions.append(english_caption)
                except:
                    pass
                break
    return captions


pool = Pool(processes=100)
results = pool.map(process_file, files)
captions = []
for result in results:
    captions.extend(result)

PICKLE_DIRECTORY = "/work/users/skoka/Data/WikipediaImageCaptionsPickles/"
import pickle
with open(PICKLE_DIRECTORY + "captions.pickle", "wb") as f:
    pickle.dump(captions, f)