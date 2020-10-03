import pandas as pd
import numpy as np
from urllib.parse import urlparse
import io
import gc
import re
import string
from utils import *

def process_data(data_file,key_covid,username_dict,standardize_dict,train_d=True):
    if train_d:
        thresh=5
        username_dict={}
        for feats in ["username"]:
            user_counts=data_file[feats].value_counts()
            user_counts = user_counts[user_counts>=thresh]
            user_dict = create_dict(user_counts.index,start=1)
            data_file[feats] = data_file[feats].map(user_dict)
            data_file[feats] = data_file[feats].fillna(0)
            username_dict[feats]=user_dict
    else:
        data_file["username"] = data_file["username"].map(username_dict["username"])
        data_file["username"] = data_file["username"].fillna(0)
    
    
    data_file["followers_null_ind"]= data_file["#followers"].isnull().astype(int)
    data_file["friends_null_ind"]= data_file["#friends"].isnull().astype(int)
    
    data_file["entity_null"] = (data_file["entities"]=="null;").astype(int)
    data_file["hashtags_null"] = (data_file["hashtags"]=="null;").astype(int)
    data_file["urls_null"] = (data_file["urls"]=="null;").astype(int)
    data_file["mentions_null"] = (data_file["mentions"]=="null;").astype(int)
      
    data_file["keyword_entities"] = data_file["entities"].apply(lambda x: keyword_entities(x,key_covid))
    data_file["keyword_hashtags"] = data_file["hashtags"].apply(lambda x: keyword_hashtags(x,key_covid))
        
    data_file["no_entities"] = data_file["entities"].apply(lambda x: count(x,";"))
    data_file["no_urls"] = data_file["urls"].apply(lambda x: count(x,":-:"))
    data_file["no_hashtags"] = data_file["hashtags"].apply(lambda x: count(x," "))
    data_file["no_mentions"] = data_file["mentions"].apply(lambda x: count(x," "))
    data_file["unique_hashtags"] = data_file["hashtags"].apply(lambda x: unique_hashtags(x))
    data_file["hashtags_char"] = data_file["hashtags"].apply(lambda x: count_words(x))
    
    data_file["sentiment_encoded"] = data_file["sentiment"].apply(lambda x: one_hot_sentiment(x))
    #data_file["sentiment_overall"] = data_file["sentiment"].apply(lambda x: overall_senti(x))
    
    data_file["week"] = data_file["timestamp"].apply(lambda x: x.split(" ")[0])
    #data_file["month"] = data_file["timestamp"].apply(lambda x: x.split(" ")[1])
    data_file["day"] = data_file["timestamp"].apply(lambda x: x.split(" ")[2])
    data_file["time"] = data_file["timestamp"].apply(lambda x: x.split(" ")[3])
    data_file["year"] = data_file["timestamp"].apply(lambda x: x.split(" ")[5])
    
    if train_d:
        username_dict["week"] = create_dict(data_file["week"].unique())
    
    data_file["week"] = data_file["week"].apply(lambda x: one_hot_week(x,username_dict["week"]))
    #data_file["month"] = data_file["month"].map(month_dict)
    data_file["time"] = data_file["time"].apply(lambda x: conv_dtime(x))
    data_file["day"] = data_file["day"].astype(int)
    data_file["year"] = data_file["year"].map({"2019":0,"2020":1})
    
    data_file["follow/friends"]=data_file["#followers"].astype(float)/(data_file["#friends"].astype(float)+1)
    data_file["friends/favorites"]=data_file["#friends"].astype(float)/(data_file["#favorites"].astype(float)+1)
    data_file["favorites/follow"]=data_file["#favorites"].astype(float)/(data_file["#followers"].astype(float)+1)
   
    data_file["#followers"] = np.log(data_file["#followers"].astype(int)+1)
    data_file["#friends"] = np.log(data_file["#friends"].astype(int)+1)
    data_file["#favorites"] = np.log(data_file["#favorites"].astype(int)+1)
    data_file["no_entities"] = np.log(data_file["no_entities"].astype(int)+1)
    data_file["no_urls"] = np.log(data_file["no_urls"].astype(int)+1)
    data_file["no_mentions"] = np.log(data_file["no_mentions"].astype(int)+1)
    data_file["no_hashtags"]=np.log(data_file.no_hashtags+1)
    #data_file["month"] = np.log(data_file["month"].astype(int)+1)
    data_file["day"] = np.log(data_file["day"].astype(int)+1)
    data_file["time"] = np.log(data_file["time"].astype(int)+1)
    data_file["follow/friends"]=np.log(data_file["follow/friends"]+1)
    data_file["friends/favorites"]=np.log(data_file["friends/favorites"]+1)
    data_file["favorites/follow"]=np.log(data_file["favorites/follow"]+1)
    data_file["unique_hashtags"] = np.log(data_file["unique_hashtags"]+1)
    data_file["hashtags_char"] = np.log(data_file["hashtags_char"]+1)
    data_file["keyword_entities"]=np.log(data_file["keyword_entities"]+1)
    data_file["keyword_hashtags"]=np.log(data_file["keyword_hashtags"]+1)
    #data_file["user_reliability"] = np.log(data_file["user_reliability"]+1)
    
    
    #standardize features
    cont_features=['#favorites', '#followers', '#friends', 'day',
       'no_entities', 'no_hashtags', 'no_mentions', 'no_urls','time',"follow/friends","friends/favorites","favorites/follow","unique_hashtags","hashtags_char","keyword_entities",
              "keyword_hashtags"]
    if train_d:
        standardize_dict={}
        for feats in cont_features:
            mu = data_file[feats].mean()
            std = data_file[feats].std()
            data_file[feats] = (data_file[feats]-mu)/std
            standardize_dict[feats] = {"mu":mu,"std":std}
    else:
        for feats in cont_features:
            metri = standardize_dict[feats]
            mu = metri["mu"]
            std = metri["std"]
            data_file[feats] = (data_file[feats]-mu)/std
    
    if train_d:
        return data_file,username_dict,standardize_dict
    else:
        return data_file