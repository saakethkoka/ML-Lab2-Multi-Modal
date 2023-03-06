# %%
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import pickle

plt.rcParams['font.size'] = 12
sns.set_style("dark")
warnings.filterwarnings('ignore')

# %% [markdown]
# # **Image Captioning**
# 
# **What is Image Captioning ?**
# - Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions.
# - This task lies at the intersection of computer vision and natural language processing. Most image captioning systems use an encoder-decoder framework, where an input image is encoded into an intermediate representation of the information in the image, and then decoded into a descriptive text sequence.
# 
# **CNNs + RNNs (LSTMs)**
# - To perform Image Captioning we will require two deep learning models combined into one for the training purpose
# - CNNs extract the features from the image of some vector size aka the vector embeddings. The size of these embeddings depend on the type of pretrained network being used for the feature extraction
# - LSTMs are used for the text generation process. The image embeddings are concatenated with the word embeddings and passed to the LSTM to generate the next word
# - For a more illustrative explanation of this architecture check the Modelling section for a picture representation

# %% [markdown]
# <img src="https://miro.medium.com/max/1400/1*6BFOIdSHlk24Z3DFEakvnQ.png">

# %%
image_path = '/work/users/skoka/Data/flickr30k_images/flickr30k_images/'

# %%
data = pd.read_csv("/work/users/skoka/Data/flickr30k_images/results.csv", delimiter='|')
data.head()
# rename data columns
data.columns = ['image', 'comment_number', 'caption']
# convert caption to string
data['caption'] = data['caption'].astype(str)

# %%
def readImage(path,img_size=224):
    img = load_img(path,color_mode='rgb',target_size=(img_size,img_size))
    img = img_to_array(img)
    img = img/255.
    
    return img

def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize = (20 , 20))
    n = 0
    for i in range(15):
        n+=1
        plt.subplot(5 , 5, n)
        plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
        image = readImage(f"/work/users/skoka/Data/flickr30k_images/flickr30k_images/{temp_df.image[i]}")
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")

# %% [markdown]
# # **Visualization**
# - Images and their corresponding captions

# %% [markdown]
# # **Caption Text Preprocessing Steps**
# - Convert sentences into lowercase
# - Remove special characters and numbers present in the text
# - Remove extra spaces
# - Remove single characters
# - Add a starting and an ending tag to the sentences to indicate the beginning and the ending of a sentence
# 
# <img src='http://zjpnrm2br14wspo448nls17u-wpengine.netdna-ssl.com/wp-content/uploads/2020/09/processing-steps.png' >

# %%
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]",""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+"," "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word)>1]))
    data['caption'] = "startseq "+data['caption']+" endseq"
    return data

# %% [markdown]
# ## __Preprocessed Text__

# %%
data = text_preprocessing(data)
captions = data['caption'].tolist()
captions[:10]

# %% [markdown]
# ## __Tokenization and Encoded Representation__
# - The words in a sentence are separated/tokenized and encoded in a one hot representation
# - These encodings are then passed to the embeddings layer to generate word embeddings
# 
# <img src='https://lena-voita.github.io/resources/lectures/word_emb/lookup_table.gif'>

# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

images = data['image'].unique().tolist()
nimages = len(images)

split_index = round(0.85*nimages)
train_images = images[:split_index]
val_images = images[split_index:]

train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]

train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)

tokenizer.texts_to_sequences([captions[1]])[0]

# %% [markdown]
# # **Image Feature Extraction**
# - DenseNet 201 Architecture is used to extract the features from the images
# - Any other pretrained architecture can also be used for extracting features from these images
# - Since the Global Average Pooling layer is selected as the final layer of the DenseNet201 model for our feature extraction, our image embeddings will be a vector of size 1920
# 
# <img src="https://imgur.com/wWHWbQt.jpg">

# %%
# model = DenseNet201()
# fe = Model(inputs=model.input, outputs=model.layers[-2].output)

# img_size = 224
# features = {}

# # Get unique image list
# image_list = data['image'].unique().tolist()

# # Batch images in blocks of 32
# for i in tqdm(range(0, len(image_list), 32)):
#     batch_images = image_list[i:i+32]
#     batch_features = {}
#     for image in batch_images:
#         img = load_img(os.path.join(image_path,image),target_size=(img_size,img_size))
#         img = img_to_array(img)
#         img = img/255.
#         img = np.expand_dims(img,axis=0)
#         batch_features[image] = img

#     batch_features = fe.predict(np.concatenate(list(batch_features.values()), axis=0), verbose=0)
#     for j, image in enumerate(batch_images):
#         features[image] = batch_features[j]

# %%
# # save features to pickle file
# with open("/work/users/skoka/Data/flicker30k_pickles/features.pkl", "wb") as f:
#     pickle.dump(features, f)

# %%
# read pickle file:
with open("/work/users/skoka/Data/flicker30k_pickles/features.pkl", "rb") as f:
    features = pickle.load(f)

# %% [markdown]
# # **Data Generation**
# - Since Image Caption model training like any other neural network training is a highly resource utillizing process we cannot load the data into the main memory all at once, and hence we need to generate the data in the required format batch wise
# - The inputs will be the image embeddings and their corresonding caption text embeddings for the training process
# - The text embeddings are passed word by word for the caption generation during inference time

# %%
class CustomDataGenerator(Sequence):
    
    def __init__(self, df, X_col, y_col, batch_size, directory, tokenizer, 
                 vocab_size, max_length, features,shuffle=True):
    
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.directory = directory
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __getitem__(self,index):
    
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
        X1, X2, y = self.__get_data(batch)
        return (X1, X2), y
    
    def __get_data(self,batch):
        
        X1, X2, y = list(), list(), list()
        
        images = batch[self.X_col].tolist()
           
        for image in images:
            feature = self.features[image]
            
            captions = batch.loc[batch[self.X_col]==image, self.y_col].tolist()
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1,len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
        return X1, X2, y

# %% [markdown]
# # **Modelling**
# - The image embedding representations are concatenated with the first word of sentence ie. starseq and passed to the LSTM network 
# - The LSTM network starts generating words after each input thus forming a sentence at the end

# %% [markdown]
# <img src='https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/03-advanced/image_captioning/png/model.png'>

# %%
input1 = Input(shape=(1920,))
input2 = Input(shape=(max_length,))

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
from tensorflow.keras.utils import plot_model

# %%
# load model from file
# import load_model
from tensorflow.keras.models import load_model
model = load_model('model.h5')

# %% [markdown]
# ## **Model Modification**
# - A slight change has been made in the original model architecture to push the performance. The image feature embeddings are added to the output of the LSTMs and then passed on to the fully connected layers
# - This slightly improves the performance of the model orignally proposed back in 2014: __Show and Tell: A Neural Image Caption Generator__ (https://arxiv.org/pdf/1411.4555.pdf)

# %%
plot_model(caption_model)

# %%
caption_model.summary()

# %%
train_generator = CustomDataGenerator(df=train,X_col='image',y_col='caption',batch_size=64,directory=image_path,
                                      tokenizer=tokenizer,vocab_size=vocab_size,max_length=max_length,features=features)

validation_generator = CustomDataGenerator(df=test,X_col='image',y_col='caption',batch_size=64,directory=image_path,
                                      tokenizer=tokenizer,vocab_size=vocab_size,max_length=max_length,features=features)

# %%
[X1, X2], y = train_generator.__getitem__(0)

# %%
# Predict cap

# %%
model_name = "model-from-kaggle.h5"
checkpoint = ModelCheckpoint(model_name,
                            monitor="val_loss",
                            mode="min",
                            save_best_only = True,
                            verbose=1)

earlystopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00000001)

# %% [markdown]
# ## **Let's train the Model !**
# 
# <img src='https://miro.medium.com/max/1400/1*xIXqf46yYonSXkUOWcOCvg.gif'>

# %%
history = caption_model.fit_generator(
        train_generator,
        epochs=150,
        validation_data=validation_generator,
        callbacks=[checkpoint,learning_rate_reduction], verbose=2)

# %%
# __get_data
(X1, X2), y = train_generator.__getitem__(0)

# %%
X1.shape, X2.shape, y.shape

# %% [markdown]
# # **Inference**
# - Learning Curve (Loss Curve)
# - Assessment of Generated Captions (by checking the relevance of the caption with respect to the image, BLEU Score will not be used in this kernel)

# %% [markdown]
# ## **Learning Curve**
# - The model has clearly overfit, possibly due to less amount of data
# - We can tackle this problem in two ways
#     1. Train the model on a larger dataset Flickr40k
#     2. Attention Models

# %%
plt.figure(figsize=(20,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %% [markdown]
# ## **Caption Generation Utility Functions**
# - Utility functions to generate the captions of input images at the inference time.
# - Here the image embeddings are passed along with the first word, followed by which the text embedding of each new word is passed to generate the next word

# %%
def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

# %%
def predict_caption(model, image, tokenizer, max_length, features):
    
    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        # reshape feature to 1 sample
        feature = feature.reshape((1,1920))

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
            
        in_text+= " " + word
        
        if word == 'endseq':
            break
            
    return in_text 

# %%

index = 34
pred = predict_caption(model, list(features.keys())[index], tokenizer, max_length, features)
print(pred)

# print real captions for the same image:
print(train.loc[train['image']==list(features.keys())[index],'caption'].tolist())




# %% [markdown]
# ## **Taking 15 Random Samples for Caption Prediction**

# %%
samples = test.sample(15)
samples.reset_index(drop=True,inplace=True)

# %%
for index,record in samples.iterrows():

    img = load_img(os.path.join(image_path,record['image']),target_size=(224,224))
    img = img_to_array(img)
    img = img/255.
    
    caption = predict_caption(model, record['image'], tokenizer, max_length, features)
    samples.loc[index,'caption'] = caption

# %% [markdown]
# # **Results**
# - As we can clearly see there is some redundant caption generation e.g. Dog running through the water, overusage of blue shirt for any other coloured cloth
# - The model performance can be further improved by training on more data and using attention mechanism so that our model can focus on relevant areas during the text generation
# - We can also leverage the interprettability of the attention mechanism to understand which areas of the image leads to the generation of which word

# %%
display_images(samples)

# %% [markdown]
# <p style='font-size: 18px'><strong>Conclusion: </strong>This may not be the best performing model, but the objective of this kernel is to give a gist of how Image Captioning problems can be approached. In the future work of this kernel <strong>Attention model</strong> training and <strong>BLEU Score</strong> assessment will be performed.</p>


