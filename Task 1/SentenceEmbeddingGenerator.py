#####################################################
# SentenceEmbeddingGenerator.py
# 
# This program is use to generate text features for the following program:
# VQAClassifier_baseline.py
#
# If you already have file "vaq2.cmp9137.sentence_transformers.txt", you
# DON'T need to generate them again. Feel free to re-run it, or to change
# it if you wish to try out alternative versions of sentence embeddings. 
# 
# To run this Python program you will need to install the following: 
# pip install sentence-transformers
# 
# Version: 1.0 
# Date: 01 March 2023
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

IMAGES_PATH = r"C:\\Lincoln\\data\\VAQ2.0\\val2014-resised"
whole_data_file = IMAGES_PATH+"\\..\\vaq2.0.token.txt"
train_data_file = IMAGES_PATH+"\\..\\vaq2.0.TrainImages.txt"
dev_data_file = IMAGES_PATH+"\\..\\vaq2.0.DevImages.txt"
test_data_file = IMAGES_PATH+"\\..\\vaq2.0.TestImages.txt"
output_file = "vaq2.cmp9137.sentence_transformers.txt"

files2read = [train_data_file, dev_data_file, test_data_file]
sentence_embeddings = {}

for file_name in files2read:
        print("READING "+str(file_name))
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_name, text = line.split("\t")
                text = text.replace("? yes", "?")
                text = text.replace("? no", "?")
                text_embedding = model.encode([text])[0]
                if text not in sentence_embeddings:
                    sentence_embeddings[text] = text_embedding

                if len(sentence_embeddings)%100 == 0:
                    print("|sentence_embeddings|="+str(len(sentence_embeddings)))

print("SAVING file in "+str(output_file))
sef = open(output_file, 'wb')
pickle.dump(sentence_embeddings, sef)
f.close()
print("Done!")