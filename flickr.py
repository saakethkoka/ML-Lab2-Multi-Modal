# %%
import pandas as pd
import numpy as np
from PIL import Image
import os
import pickle
from IPython.display import Image as IPyImage
from IPython.display import display
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pydot

# %% [markdown]
# ### Constants

# %%
IMAGES_DIRECTORY = "/work/users/skoka/Data/flickr30k_images/flickr30k_images/"
LABELS_FILE = "/work/users/skoka/Data/flickr30k_images/results.csv"
PICKLE_DIRECTORY = "/work/users/skoka/Data/flicker30k_pickles/"
GET_FROM_PICKLE = True

# %%
# reading in the labels
labels = pd.read_csv(LABELS_FILE, delimiter='|')
# convert labels[ ' comment' ] to strings
labels[' comment'] = labels[' comment'].astype(str)

# list of .jpg files from the directory
image_files = [f for f in os.listdir(IMAGES_DIRECTORY) if f.endswith('.jpg')]

# %%
# Average number of words in a sentence
avg_words = 0
for i in range(len(labels)):
    avg_words += len(labels[' comment'][i].split())
avg_words = avg_words / len(labels)
print("Average number of words in a sentence: ", avg_words)

# %%
# Converts a jpg file to a numpy array
def _read_jpg(filename):
    im = Image.open(filename)
    # resize to 224x224
    im = im.resize((224, 224))
    return np.array(im)

def process_files(files):
    images = {
        # "filename_without_extension" : numpy_array_of_image
    }
    num_read = 0
    for f in files:
        image_name = f.split('.')[0]
        image = _read_jpg(IMAGES_DIRECTORY + f)
        images[image_name] = {
            "image": _read_jpg(IMAGES_DIRECTORY + f),
            "captions": labels[labels.image_name == (image_name + ".jpg")][" comment"].tolist()
            }
        num_read += 1
        if num_read % 1000 == 0:
            print("Read {} files".format(num_read))
    return images
if not GET_FROM_PICKLE:
    images = process_files(image_files)
    with open(PICKLE_DIRECTORY + "flicker_images.pkl", "wb") as f:
        pickle.dump(images, f)
else:
    with open(PICKLE_DIRECTORY + "flicker_images.pkl", "rb") as f:
        images = pickle.load(f)

# %% [markdown]
# ### Data Formatting
# {
#     "image_name" : {
#         "image" : "image", // Numpy array of the image
#         "captions" :  [
#             "caption 1",
#             "caption 2",
#             ...
#         ]
#     }
# }

# %%
from keras.preprocessing.text import Tokenizer
captions = labels[" comment"].to_numpy()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
tokenizer.fit_on_texts(['staaaart', 'endddd'])
vocab_size = len(tokenizer.word_index) + 1
# tokenizer.word_index['staaaart'] = vocab_size -1
# tokenizer.word_index['endddd'] = vocab_size
word_index = tokenizer.word_index

# %%
len(word_index)

# %%

with open("/work/users/skoka/Data/flicker30k_pickles/features.pkl", "rb") as f:
    features = pickle.load(f)

# %%
MAX_CAPTION_LENGTH = 20
def clean_caption(caption):
    caption = caption.lower()
    # remove non alphanumeric characters
    caption = re.sub(r'[^a-zA-Z0-9\s]', '', caption)
    # add start and end tokens
    caption = 'staaaart ' + caption + ' endddd'
    return caption

images_passed_in = set()


def data_generator(batch_size=32):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for image_name, image_data in images.items():
            # add image_name to images_passed_in if not already there
            if image_name not in images_passed_in:
                images_passed_in.add(image_name)
            # image = image_data["image"]
            image = features[image_name + ".jpg"]
            captions = image_data["captions"]
            for caption in captions:
                caption = clean_caption(caption)
                caption = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(caption)):
                    in_seq, out_seq = caption[:i], caption[i]
                    in_seq = pad_sequences([in_seq], maxlen=MAX_CAPTION_LENGTH)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(image)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        X1 = np.array(X1)
                        X1 = X1 / 255
                        X2 = np.array(X2)
                        y = np.array(y)
                        yield [X1, X2], y
                        X1, X2, y = [], [], []
                        n = 0

# %%
# index = 134
# image = X1[index]
# # turn image into an Image object
# image = Image.fromarray(image.astype('uint8'), 'RGB')
# # print caption for the image, X2[0] is the caption
# caption = tokenizer.sequences_to_texts([X2[index]])[0]
# print(caption)
# yhat = np.argmax(y[index])
# word = tokenizer.index_word[yhat]
# print(word)
# image.show()

# %%
def load_embeddings(filename, embed_size):
    # the embed size should match the file you load glove from
    embeddings_index = {}
    f = open(filename)
    # save key/array pairs of the embeddings
    #  the key of the dictionary is the word, the array is the embedding
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # now fill in the matrix, using the ordering from the
    #  keras word tokenizer from before
    found_words = 0
    embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be ALL-ZEROS
            embedding_matrix[i] = embedding_vector
            found_words = found_words+1

    print("Embedding Shape:",embedding_matrix.shape, "\n",
        "Total words found:",found_words, "\n",
        "Percentage:",100*found_words/embedding_matrix.shape[0])
    return embedding_matrix

# embedding_matrix = load_embeddings("/users/skoka/Documents/ML-Lab2-Multi-Modal/numberbatch-en-19.08.txt", 300)

# %%
# Build a VGG model that takes input of size 250x250x3
# vgg = tf.keras.applications.VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# # remove the last layer of the VGG model
# vgg = tf.keras.Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
# vgg.trainable = False

image_input = tf.keras.layers.Input(shape=(1920,))


image_dense = tf.keras.layers.Flatten()(image_input)
image_dense = tf.keras.layers.Dense(256, activation='relu')(image_dense)
image_reshaped = tf.keras.layers.Reshape((1, 256))(image_dense)

caption_input = tf.keras.layers.Input(shape=(MAX_CAPTION_LENGTH,))
caption_embedding = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=False)(caption_input)

# Merge the two models
decoder_add = tf.keras.layers.concatenate([image_reshaped, caption_embedding], axis=1)

# bi directonal LSTM
decoder_lstm = tf.keras.layers.LSTM(256)(decoder_add)
x = tf.keras.layers.add([image_dense, decoder_lstm])
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)

model = tf.keras.Model(inputs=[image_input, caption_input], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot model:
model.summary()

# %%
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Reshape, concatenate, add
from keras.models import Model

input1 = Input(shape=(1920,))
input2 = Input(shape=(MAX_CAPTION_LENGTH,))

img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
merged = concatenate([img_features_reshaped,sentence_features],axis=1)
sentence_features = LSTM(256)(merged)
x = Dropout(0.5)(sentence_features)
x = add([x, img_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model(inputs=[input1,input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy',optimizer='adam')

# %%
# model save callback:
path = "model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(path, monitor='loss', save_best_only=True, mode='min')


# %%
history = caption_model.fit_generator(data_generator(batch_size=64),
                                epochs=90,
                                steps_per_epoch=50000,
                                callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.00000001, verbose=2), checkpoint],
                                verbose=2)

# %%
def predict_caption(model, image):
    in_text = 'staaaart'
    for i in range(MAX_CAPTION_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_CAPTION_LENGTH)
        yhat = model.predict([image, sequence], verbose=0)
        # print top 5 predicted words:
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        in_text += ' ' + word
        if word == 'endddd':
            break
    return in_text

# %%
model_path = "/users/skoka/Documents/ML-Lab2-Multi-Modal/model.h5"
model = tf.keras.models.load_model(model_path)

# %%
# Predict on a random image
import random
image_name = random.choice(list(images.keys()))
image = features[image_name + ".jpg"]
caption = images[image_name]["captions"][0]
print("Actual Caption:", caption)
predicted_caption = predict_caption(caption_model, image.reshape(1, 1920))
print("Predicted Caption:", predicted_caption)

# %%
len(images_passed_in)

# %%



