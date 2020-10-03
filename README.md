# CIKM2020
 This is the 1st Place solution of CIKM2020 Analyticup COVID19 Retweet Prediction Challenge - https://competitions.codalab.org/competitions/25276.
 
# MODEL ARCHITECTURE:


# MODEL PERFORMANCE



# HOW TO RUN

python main.py [data_path] [external data indicator] [path to external data] [path to the pretrained embedding file] [pretrained embedding name] [embedding size] [path to save model] [path to save test result] [use _CNN]. 

Here is an example

python test_file.py /projects/CIKM/ 1 /projects/CIKM/ /projects/CIKM/embeddings/glove.6B.300d.txt glove.840B.300d.txt 300 ./lstm_glove_model.ckpt ./test.predict test.data 0

External dataset 



# DEPENDENCIES
Python 3.6
Tensorflow 2.2.o

# REFERENCE
