import pandas as pd
import numpy as np
from urllib.parse import urlparse
import io
import sys
import gc
import re
import string
from utils import *
from batch_generator import *
from load_embedding import *
from preprocess import *
from read_n3files import *
from model import *

def main(args):
    data_path = args[0]
    is_external_dataset=int(args[1])
    external_datapath=args[2]
    embedding_path=args[3]
    wordvec_name=args[4]
    wordvec_dim=int(args[5])
    model_path=args[6]
    test_out_path=args[7]
    predict_filename = args[8]
    use_CNN= int(args[9])
    #read all the train data files
    train_data = read_data(data_path+"train.data")
    train_data["train_indicator"]=1
    train_data["retweets"] = np.log(np.array(create_labels(data_path+"train.solution"))+1)
    
    if is_external_dataset==1:
        #read march and april external dataset
        print("extract april data from n3 file")
        extract_dataset(external_datapath+"month_2020-04.n3",external_datapath+"april_external_dataset.csv")
        april_data = pd.read_csv(data_path+"april_external_dataset.csv")
        april_data["train_indicator"]=0
        april_data["retweets"] = np.log(april_data["retweets"]+1)

        print("extracting march data from n3 file")
        extract_dataset(external_datapath+"month_2020-03.n3",external_datapath+"march_external_dataset.csv")
        march_data = pd.read_csv(data_path+"march_external_dataset.csv")
        march_data["train_indicator"]=0
        march_data["retweets"] = np.log(march_data["retweets"]+1)

        train_data = pd.concat([train_data,april_data,march_data])
        del april_data
        del val_data
        gc.collect()

    #read test dataset
    test_data = read_data(data_path+predict_filename,test_d=True)
    test_data["tweet_id"]=12345
    #tes_data["train_indicator"]=0


    #train_data["username_cp"]=train_data["username"]
    #chronologically arrange data based on date
    encodeMonth={"Jan":'1',"Feb":'2',"Mar":'3',"Apr":'4',"May":'5',"Jun":'6',"Sep":'9',"Oct":'10',"Nov":'11',"Dec":'12'}
    def conv_timestamp(time):
        k = time.split(" ")
        dt=k[2]+"-"+encodeMonth[k[1]]+"-"+k[-1]+" "+k[3]
        #dt=pd.to_datetime(dt,format="%d-%m-%Y %H:%M:%S")
        return dt
    train_data["date"]=train_data["timestamp"].apply(lambda x: conv_timestamp(x))
    train_data["date"]=pd.to_datetime(train_data["date"],format="%d-%m-%Y %H:%M:%S")
    train_data=train_data.sort_values("date").reset_index(drop=True)
    train_data["month"] = train_data["date"].dt.month
    if is_external_dataset==1:
        #split train and validation sets
        other_data = train_data[train_data["train_indicator"]==0]
        train_data = train_data[train_data["train_indicator"]==1]
    ratio = 0.05
    train_size=int(train_data.shape[0]*ratio)
    train = train_data.iloc[:-train_size,:]
    valid = train_data.iloc[-train_size:,:]
    if is_external_dataset==1:
        train_data = pd.concat([train,other_data])
    train_data = train
    del train
    gc.collect()

    
    
    #read stop words for text processing
    stop_w = open("./stopwords.txt")
    stop_set =set()
    try:
        while stop_w:
                wrd = stop_w.__next__().split()[0]
                stop_set.add(wrd)
    except:
        pass
    stop_set.add("com")
    stop_set.add("org")

    #read given covid related keywords for covid keyword count feature creation
    keywords = open("./covid19_keywords.txt")
    covid_keywords=keywords.readlines()
    key_covid=[]
    for k in covid_keywords:
        key_covid+=k.split()
    key_covid = set(key_covid)

    #preprocess train
    #keep username_dict and standardize_dict empty only for the training dataset
    train_data,username_dict,standardize_dict = process_data(train_data,key_covid,{},{},train_d=True)
    #preprocess valset
    valid= process_data(valid,key_covid,username_dict,standardize_dict,train_d=False)
    #preprocess test_set
    test_data= process_data(test_data,key_covid,username_dict,standardize_dict,train_d=False)

    #create tokenizer
    tokenizer,vocab_len,word_vec = create_embeddings(train_data,embedding_path,wordvec_name,stop_set,wordvec_dim)
    #built model
    model_cp = model(username_dict,word_vec,wordvec_dim,vocab_len,use_CNN)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_msle_loss', 
        verbose=1,
        patience=5,
        mode='min',
        restore_best_weights=True)

    checkpoint_path= model_path
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,monitor='val_msle_loss',
                                                     save_best_only=True,
                                                     save_weights_only=True,mode='min',
                                                     verbose=1)
    #set batch sizes
    batch_train = batch_generator(train_data,tokenizer,stop_set,batch_size=2048,shuffle=True,is_train=True)
    batch_valid = batch_generator(valid,tokenizer,stop_set,batch_size=2048,shuffle=False,is_train=True)
    test_batch = batch_generator(test_data,tokenizer,stop_set,batch_size=2048,shuffle=False,is_train=False)

    #train the model
    model_cp.fit(batch_train,epochs=50, verbose=1,validation_data = batch_valid,callbacks =[early_stopping,cp_callback])

    #predict on test dataset
    test_results = model_cp.predict(test_batch)
    #transform the results back to original form and round it
    test_results = np.exp(test_results)-1
    test_results = np.round(test_results)

    write_output(test_results,test_out_path)

if __name__ == "__main__":
    main(sys.argv[1:])
    
    
