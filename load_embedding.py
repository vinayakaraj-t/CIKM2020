import pandas as pd
import numpy as np
from urllib.parse import urlparse
import io
import gc
import re
import string
from utils import *
import tensorflow as tf

def load_vectors(fname,count_words):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        data_list=[]
        for line in fin:
            tokens = line.rstrip().split(' ')
            tk = tokens[0]
            if tk in count_words:
                vec=list(map(float, tokens[1:]))
                data[tk] = vec
                data_list.append(vec)
        return data,data_list
    
    
def glove_load_vectors(fname,count_words):
        data={}
        fastvec = open(fname)
        counter=1
        data_list=[]
        while counter>0:
            try:
                f=fastvec.__next__()
                tokens = f.rstrip().split(' ')
                tk=tokens[0]
                if tk in count_words:
                    vec = list(map(float, tokens[1:]))
                    data[tk] = vec
                    data_list.append(vec)
                counter+=1
            except:
                print("total tokens",counter)
                counter=0
                pass
        return data,data_list

def create_embeddings(train_data,embedding_path,wordvec_name,stop_set,word_dim):

    entity1 =  train_data["entities"].apply(lambda x: combine_entity(x))
    mention_dt =  train_data["hashtags"].apply(lambda x: hashtag(x))
    url_dt1 =  train_data["urls"].apply(lambda x: process_urlPath(x,0,stop_set))
    url_dt2 =  train_data["urls"].apply(lambda x: process_urlPath(x,1,stop_set))
    mention_splt = train_data["mentions"].apply(lambda x: hashtag(x))
    
    dt_concat =pd.concat([entity1,mention_dt,url_dt1,url_dt2,mention_splt],axis=0)
    
    print("create entity tokenizer")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None)

    #tokenizer.fit_on_texts(pd.concat([entity1,mention_dt,url_dt,mention_splt],axis=0))
    tokenizer.fit_on_texts(dt_concat)
    
    count_thres = 15
    count_words = {w:c for w,c in tokenizer.word_counts.items() if c >= count_thres}

    word_counts= len(count_words)+1#one for oov and one for less count words

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=word_counts,
        filters='',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None)

    #tokenizer.fit_on_texts(pd.concat([entity1,mention_dt,url_dt,mention_splt],axis=0))
    tokenizer.fit_on_texts(dt_concat)
    
    print("load embedding vectors")
    if wordvec_name.split(".")[0]=="glove":
        fastvec,fastvec_list = glove_load_vectors(embedding_path,count_words)
    else:
        fastvec,fastvec_list = load_vectors(embedding_path,count_words)

    cand=np.array(fastvec_list,dtype='float32')
    mu=np.mean(cand, axis=0)
    Sigma=np.cov(cand.T)
    norm=np.random.multivariate_normal(mu, Sigma, 1)
    norm = list(np.reshape(norm, word_dim))

    word_counts = len(count_words)+1
    word_vectors = np.zeros((word_counts,word_dim))
    id_w = tokenizer.index_word

    for k in range(1,word_vectors.shape[0]):
        ky = id_w[k]
        if ky in fastvec:
            word_vectors[k,:]=fastvec[ky]
        else:
            word_vectors[k,:]= norm
    
    return tokenizer,word_counts,word_vectors
