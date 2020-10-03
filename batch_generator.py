from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import *

class batch_generator(Sequence):
    def __init__(self, dataset,tokenizer,stop_set,batch_size,shuffle=True,is_train=True):
        self.dataset=dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.is_train = is_train
        self.stop_set=stop_set
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dataset.shape[0] / float(self.batch_size)))
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.dataset.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        data_chunck = self.dataset.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        username_ = data_chunck[["username"]].values
        sentiment_encode = np.array(data_chunck["sentiment_encoded"].tolist())
        week_encode = np.array(data_chunck["week"].tolist())
        other_f = ['#favorites', '#followers', '#friends', 'day',
      'no_entities', 'no_hashtags', 'no_mentions', 'no_urls','time',"year","follow/friends","friends/favorites","favorites/follow","unique_hashtags","hashtags_char","entity_null","hashtags_null","urls_null","mentions_null","keyword_entities","keyword_hashtags",
                  'followers_null_ind', 'friends_null_ind']
        all_fea= data_chunck[other_f].values
        
        entity1 =  data_chunck["entities"].apply(lambda x: combine_entity(x))
        entity_sequences1 = self.tokenizer.texts_to_sequences(entity1)
        entity_pad1 = tf.keras.preprocessing.sequence.pad_sequences(
            entity_sequences1, maxlen=10, dtype='int32', padding='pre', truncating='post')
        
        hashtag_process =  data_chunck["hashtags"].apply(lambda x: hashtag(x))
        valid_hashtag = self.tokenizer.texts_to_sequences(hashtag_process)
        hashtag_valid = tf.keras.preprocessing.sequence.pad_sequences(
            valid_hashtag, maxlen=5, dtype='int32', padding='pre', truncating='post')
        
        url_dt1 =  data_chunck["urls"].apply(lambda x: process_urlPath(x,0,self.stop_set))
        urlPath_sequences1 = self.tokenizer.texts_to_sequences(url_dt1)
        urlPath_valid1 = tf.keras.preprocessing.sequence.pad_sequences(
            urlPath_sequences1, maxlen=3, dtype='int32', padding='pre', truncating='post')

        url_dt2 =  data_chunck["urls"].apply(lambda x: process_urlPath(x,1,self.stop_set))
        urlPath_sequences2 = self.tokenizer.texts_to_sequences(url_dt2)
        urlPath_valid2 = tf.keras.preprocessing.sequence.pad_sequences(
            urlPath_sequences2, maxlen=15, dtype='int32', padding='pre', truncating='post')

        mention_splt =  data_chunck["mentions"].apply(lambda x: hashtag(x))
        mention_validsplt = self.tokenizer.texts_to_sequences(mention_splt)
        mention_validsplt = tf.keras.preprocessing.sequence.pad_sequences(
            mention_validsplt, maxlen=5, dtype='int32', padding='pre', truncating='post')
        
        batch_x = [username_,sentiment_encode,week_encode,all_fea,entity_pad1,hashtag_valid,urlPath_valid1,urlPath_valid2,mention_validsplt]                
        if self.is_train:
            batch_y = data_chunck["retweets"]
            return batch_x,batch_y
        else:
            return batch_x