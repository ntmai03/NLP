# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 06:06:48 2018

@author: DELL
"""

# import data set
filename ='D:\\OE\\SA_01.txt'
names = ['text','label']
df = pd.read_csv(filename,header=None,sep='\t',quoting =3)
print(df.head())

words_set = Set()
for review in training_set:
    score = review['score']
    text = review['review_text']
    splitted_text = split_text(text)
    for word in splitted_text:
        if word not in words_set:
            words_set.add(word)
            BOW_df.loc[word] = [0,0,0,0,0]
            BOW_df.ix[word][score] += 1
        else:
            BOW_df.ix[word][score] += 1
