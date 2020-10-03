import re
def extract_values(mid):
    text0="" #tweetid
    text1="" #date
    text2="" #userid
    text3="" #favorites
    text4="" #retweet
    text5="" #entity
    text6="" #mention
    text7="" #hashtags
    text8="" #url
    text9="" #possenti
    text10=""#negative sent

    for o in mid:
        split_str =o.split()
        test_str = split_str[0]
        ll =re.sub("[0-9]","",test_str)

        if ll=="_:t" and split_str[2] =="sioc:Post":
            tw_id = re.sub("[^0-9]","",test_str)
            text0+=tw_id
            date = re.sub("\"","",split_str[5].split("^^")[0])
            text1+=date
            user = split_str[11][3:]
            text2+=user

        if ll=="_:i_":
            if split_str[5] =='schema:LikeAction':
                favorite = re.sub("\"","",split_str[-2].split("^^")[0])
                text3+=favorite
            elif split_str[5] =='schema:ShareAction':
                retweet = re.sub("\"","",split_str[-2].split("^^")[0])
                text4+=retweet

        if ll =='_:emPos':
            pos_senti = re.sub("\"","",split_str[5].split("^^")[0])
            text9+=pos_senti

        if ll =='_:emNeg':
            neg_senti = re.sub("\"","",split_str[5].split("^^")[0])
            text10+=neg_senti

        if ll=="_:t" and split_str[1] =='schema:citation':
            text8+= split_str[2][1:-1]+":-:"

        if ll=="_:m_":
            text6+=split_str[-2][1:-1]+":"

        if ll=="_:e_":
            spl = o.split("\"")
            entity = spl[1]+":"+spl[2].split("<")[1].split(">")[0].split("/")[-1]+":"+spl[3]+";"
            text5+=entity
        if ll=='_:h_':
            text7+= split_str[-2][1:-1]+":"
    return text0+"\t"+text1+"\t"+text2+"\t"+text3+"\t"+text4+"\t"+text5+"\t"+text6+"\t"+text7+"\t"+text8+"\t"+text9+"\t"+text10

def extract_dataset(filename,save_filepath):
    file=open(filename)
    count=0
    str_=11
    mid=[]
    result=open(save_filepath,"w")
    while str_>0:
        try:
            o=file.__next__()
            if count>9:
                split_str=o.split()
                if len(split_str)>0:
                    mid.append(o)
                else:
                    values= extract_values(mid)
                    result.write(values+"\n")
                    mid=[]
            count+=1
            if count%500000==0:
                print(count)
        except:
            str_=0
            result.close()
