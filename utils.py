import pandas as pd
import numpy as np
from urllib.parse import urlparse
import io
import gc
import re
import string

def read_data(data_file,test_d=False):
    data = open(data_file)
    
    try:
        train_data=[]
        while data:
            z=data.__next__().split("\t")
            z[-1]=z[-1].split()[0]
            train_data.append(z)
    except:
        pass
    
    train_data= pd.DataFrame(train_data)
    
    if test_d:
        train_data.columns = ["username","timestamp","#followers","#friends","#favorites","entities","sentiment","mentions","hashtags","urls"]
    else:
        train_data.columns = ["tweet_id","username","timestamp","#followers","#friends","#favorites","entities","sentiment","mentions","hashtags","urls"]
    return train_data


def create_labels(data_lbl):
    response = open(data_lbl)
    retweets=[]
    try:
        while response:
            retw = response.__next__()
            retw = int(retw)
            retweets.append(retw)
    except:
        pass
    return retweets


def conv_timestamp(time):
    k = time.split(" ")
    dt=k[2]+"-"+encodeMonth[k[1]]+"-"+k[-1]+" "+k[3]
    #dt=pd.to_datetime(dt,format="%d-%m-%Y %H:%M:%S")
    return dt

def create_dict(unique_val,start=0):
    count=start
    dic ={}
    for k in unique_val:
        dic[k]=count
        count+=1
    return dic

def keyword_entities(x,key_covid):
    if x=="null;":
        return 0
    else:
        s = x.split(";")[:-1]
        ff=[]
        for z in s:
            d = z.split(":")
            ent=[d[0].lower(),d[1].lower()]
            ff+= ent
        count=0
        for zz in ff:
            if zz in key_covid:
                count+=1
        return count
    
def keyword_hashtags(x,key_covid):
    if x=="null;":
        return 0
    else:
        s = x.split(" ")
        ff=[]
        for z in s:
            ff.append(z.lower())
        count=0
        for zz in ff:
            if zz in key_covid:
                count+=1
        return count

def count(x,sep):
    if x!="null;":
        cc = x.split(sep)
        return len(cc)
    else:
        return 0
    
def unique_hashtags(x):
    if x=="null;":
        return 0
    else:
        x=x.split(" ")
        return np.unique(x).size/len(x)
    
def count_words(x):
    if x=="null;":
        return 0
    else:
        return len(x)
    
def one_hot_sentiment(x):
    spl = x.split(" ")
    d =[0]*10
    d[int(spl[0])-1]=1
    d[int(spl[-1])]=1
    return d

def one_hot_week(x,dict_):
    len_=len(dict_)
    z=[0]*len_
    z[dict_[x]]=1
    return z

def conv_dtime(v):
        v = v.split(":")
        return (float(v[0])*3600+float(v[1])*60+float(v[2]))/3600
    
def split_url(line, part):
    # this is copy of split_url function from the URLNetrepository: https://github.com/Antimalweb/URLNet/blob/master/utils.py
    if line.startswith("http://"):
        line=line[7:]
    if line.startswith("https://"):
        line=line[8:]
    if line.startswith("ftp://"):
        line=line[6:]
    if line.startswith("www."):
        line = line[4:]
    slash_pos = line.find('/')
    if slash_pos > 0 and slash_pos < len(line)-1: # line = "fsdfsdf/sdfsdfsd"
        primarydomain = line[:slash_pos]
        path_argument = line[slash_pos+1:]
        path_argument_tokens = path_argument.split('/')
        pathtoken = "/".join(path_argument_tokens[:-1])
        last_pathtoken = path_argument_tokens[-1]
        if len(path_argument_tokens) > 2 and last_pathtoken == '':
            pathtoken = "/".join(path_argument_tokens[:-2])
            last_pathtoken = path_argument_tokens[-2]
        question_pos = last_pathtoken.find('?')
        if question_pos != -1:
            argument = last_pathtoken[question_pos+1:]
            pathtoken = pathtoken + "/" + last_pathtoken[:question_pos]     
        else:
            argument = ""
            pathtoken = pathtoken + "/" + last_pathtoken          
        last_slash_pos = pathtoken.rfind('/')
        sub_dir = pathtoken[:last_slash_pos]
        filename = pathtoken[last_slash_pos+1:]
        file_last_dot_pos = filename.rfind('.')
        if file_last_dot_pos != -1:
            file_extension = filename[file_last_dot_pos+1:]
            filename = filename[:file_last_dot_pos]
        else:
            file_extension = "" 
    elif slash_pos == 0:    # line = "/fsdfsdfsdfsdfsd"
        primarydomain = line[1:]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    elif slash_pos == len(line)-1:   # line = "fsdfsdfsdfsdfsd/"
        primarydomain = line[:-1]
        pathtoken = ""
        argument = ""
        sub_dir = ""     
        filename = ""
        file_extension = ""
    else:      # line = "fsdfsdfsdfsdfsd"
        primarydomain = line
        pathtoken = ""
        argument = ""
        sub_dir = "" 
        filename = ""
        file_extension = ""
    if part == 'pd':
        return primarydomain
    elif part == 'path':
        return pathtoken
    elif part == 'argument': 
        return argument 
    elif part == 'sub_dir': 
        return sub_dir 
    elif part == 'filename': 
        return filename 
    elif part == 'fe': 
        return file_extension
    elif part == 'others': 
        if len(argument) > 0: 
            return pathtoken + '?' +  argument 
        else: 
            return pathtoken 
    else:
        return primarydomain, pathtoken, argument, sub_dir, filename, file_extension
    
def split_word(x,pos):
    if x == "null;":
        return ""
    else:
        s = x.split(";")[:-1]
        ff=[]
        check_set=set()
        for z in s:
            d = z.split(":")
            ent=d[pos]
            ent2 = re.sub("%\d+"," ",d[pos+1])
            ent2 = re.sub("_","",ent2)
            ent2 = ent2.lower().split()
            #ent = [ent.lower()]+ent2
            ent = [ent.lower()]
            for zz in ent:
                if zz not in check_set:
                    ff.append(zz)
                    check_set.add(zz)
        rt= " ".join(ff)
        return rt

def create_bigram(h):
    dr=[]
    start=h[0]
    for k in h[1:]:
        dr.append(start+k)
        start=k
    return dr

def bigrams(x,pos):
    if x == "null;":
        return ""
    else:
        s = x.split(";")[:-1]
        ff=[]
        check_set=set()
        for z in s:
            d = z.split(":")
            ent2 = re.sub("%\d+"," ",d[pos+1])
            ent2 = re.sub("_"," ",ent2)
            ent2 = ent2.lower().split()
            if len(ent2)>1:
                ff+=create_bigram(ent2)
        if len(ff)==0:
            return ""
        else:
            return " ".join(ff)

def combine_entity(x):
    a = split_word(x,0)+" "+bigrams(x,0)
    a = " ".join(a.split())
    return a


def hashtag(k):
    if k=="null;":
        return ""
    else:
        orig=k.lower().split()
        k= re.sub(r'[^a-zA-Z]'," ",k)
        k=" ".join(k.split())
        k=" ".join([a for a in re.split('([A-Z][a-z]+)', k) if a])
        k=k.lower().split()+orig
        ff=[]
        check_set=set()
        for zz in k:
            if zz not in check_set:
                ff.append(zz)
                check_set.add(zz)
        return " ".join(ff)
    
def process_urlPath(x,pos,stop_set):
    if x=="null;":
        return ""
    else:
        spl=x.split(":-:")[:-1]
        res=[]
        for k in spl:
            uert = split_url(k,"d")
            rt = re.sub("[^A-Za-z]"," ",uert[pos])
            rt = rt.split()
            ur_spl=[]
            for ul in rt:
                if ul not in stop_set:
                    ur_spl.append(ul)
            res.append(" ".join(ur_spl))
        return " ".join(res)
    
def write_output(f,file_path):
    file = open(file_path,"w")
    mm=f.ravel()
    len_mm = len(mm)
    cc=0
    for k in mm:
        cc+=1
        if cc==len_mm:
            file.write(str(int(k)))
        else:
            file.write(str(int(k))+"\n")
    file.close()
