import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import io
import gc
import re
import string
from urllib.parse import urlparse

def attention_user(embedding_entity1,embedding_user1,use_cnn):
    user_embedding_word= tf.keras.layers.Dense(100,activation='relu')(embedding_user1)
    user_embedding_word= tf.keras.layers.Flatten()(user_embedding_word)
    if use_cnn==1:
        embedding_entity1 = tf.keras.layers.Convolution1D(filters=150, kernel_size=3,  padding='same', activation='relu', strides=1)(embedding_entity1)
    else:
        embedding_entity1=tf.keras.layers.LSTM(150,return_sequences=True)(embedding_entity1)
    attention_a=tf.keras.layers.Dot((2, 1))([embedding_entity1,tf.keras.layers.Dense(150,activation='tanh')(user_embedding_word)])
    attention_weight = tf.keras.layers.Activation('softmax')(attention_a)
    news_rep=tf.keras.layers.Dot((1, 1))([embedding_entity1, attention_weight])
    return news_rep


def msle_loss(act,pred):
    pred = tf.keras.backend.exp(pred)-1
    pred = tf.keras.backend.round(pred)
    pred = tf.keras.backend.log(pred+1)
    error = (act-pred)**2
    return tf.keras.backend.mean(error)


def model(feature_dicts,word_vec,vec_d,vocab_len,use_cnn):
    user_length = len(feature_dicts["username"])+1
    #user embedding length
    embedding_len = 64
    tf.compat.v1.disable_eager_execution()
    #create input heads
    user_inp = tf.keras.layers.Input((1,))
    sentiment_inp = tf.keras.layers.Input((10,))
    week_inp=tf.keras.layers.Input((7,))
    all_feats_inp = tf.keras.layers.Input((23,))
    entity_inp1 = tf.keras.layers.Input((10,))
    hashtag_inp = tf.keras.layers.Input((5,))
    urlPath_inp1 = tf.keras.layers.Input((3,))
    urlPath_inp2 = tf.keras.layers.Input((15,))
    mentionsplt_inp = tf.keras.layers.Input((5,))

    #create embedding for users
    user_embed = tf.keras.layers.Embedding(input_dim=user_length, output_dim=embedding_len,embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=123),input_length=1)
    entity_embed = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=vec_d,weights=[word_vec],trainable=True)


    #query embeddings
    embedding_user1 =user_embed(user_inp)
    embedding_user = tf.keras.layers.Lambda(lambda y: tf.keras.backend.squeeze(y, 1))(embedding_user1)

    embedding_entity1 =entity_embed(entity_inp1)
    embedding_entity1= tf.keras.layers.Dropout(0.25)(embedding_entity1)
    entity_features_conv1 = attention_user(embedding_entity1,embedding_user1,use_cnn)


    embedding_hashtag =entity_embed(hashtag_inp)
    embedding_hashtag= tf.keras.layers.Dropout(0.25)(embedding_hashtag)
    hashtag_features_conv = attention_user(embedding_hashtag,embedding_user1,use_cnn)

    embedding_urlPath1 =entity_embed(urlPath_inp1)
    embedding_urlPath1= tf.keras.layers.Dropout(0.25)(embedding_urlPath1)
    urlPath_features_conv1 = attention_user(embedding_urlPath1,embedding_user1,use_cnn)

    embedding_urlPath2 =entity_embed(urlPath_inp2)
    embedding_urlPath2= tf.keras.layers.Dropout(0.25)(embedding_urlPath2)
    urlPath_features_conv2 = attention_user(embedding_urlPath2,embedding_user1,use_cnn)

    embedding_mentionsplt =entity_embed(mentionsplt_inp)
    embedding_mentionsplt= tf.keras.layers.Dropout(0.25)(embedding_mentionsplt)
    mentionsplt_features_conv = attention_user(embedding_mentionsplt,embedding_user1,use_cnn)


    inp_feats = tf.keras.layers.concatenate([embedding_user, sentiment_inp,week_inp,all_feats_inp,entity_features_conv1,hashtag_features_conv,urlPath_features_conv1,urlPath_features_conv2,mentionsplt_features_conv], 1)

    fc1 = tf.keras.layers.Dense(500, activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(inp_feats)
    drop = tf.keras.layers.Dropout(rate=0.15,seed=234)(fc1)
    fc2 = tf.keras.layers.Dense(150, activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=234))(drop)
    drop1 = tf.keras.layers.Dropout(rate=0.15,seed=34)(fc2)
    out = tf.keras.layers.Dense(1,activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=146))(drop1)
    
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model_cp = tf.keras.models.Model(inputs=[user_inp,sentiment_inp,week_inp,all_feats_inp,entity_inp1,hashtag_inp,urlPath_inp1,urlPath_inp2,mentionsplt_inp],outputs=[out])
    model_cp.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=optimizer,metrics=[msle_loss])

    return model_cp