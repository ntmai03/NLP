pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 175)
pd.set_option('display.max_colwidth', 500)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Mining Twitter data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import json                         # for parsing the data
import pandas as pd                 # for data manipulation
import matplotlib.pyplot as plt     # for creating charts

tweets_data_path = 'D:\\MyProjects\\02_NLP\Data\\python_matlab.txt'
tweets_data = []
tweets_file = open(tweets_data_path,"r")

for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue    

# print the number of tweets     
print(len(tweets_data))

# structure tweets data into a pandas DataFrame
tweets = pd.DataFrame()
# Add 3 columns to the tweets
# tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
# tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
# tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)                     
tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
tweets['lang'] = list(map(lambda tweet: tweet['lang'], tweets_data))
tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))
# URL = https://twitter.com/IntelFPGA/status/1002948224381046784

#----------------------------------------------------------------------------------------#
# Data Visualization
#----------------------------------------------------------------------------------------#
# create 2 charts:
# 1st: the top 5 languages, 2nd: the top countries
tweets_by_lang = tweets['lang'].value_counts()
print(tweets_by_lang)
tweets_by_country = tweets['country'].value_counts()
print(tweets_by_country)

#----------------------------------------------------------------------------------------#
# Mining the tweets
#----------------------------------------------------------------------------------------#
# library for regular expression
import re

# define function to search a particular word in a text
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

# add 2 columns in tweets dataframe
tweets['python'] = tweets['text'].apply(lambda tweet: word_in_text('python', tweet))
tweets['matlab'] = tweets['text'].apply(lambda tweet: word_in_text('matlab', tweet))
print('Print the first 10 tweets:\n')
print(tweets.head(10))

# Make a simple comparison chart
prg_langs = ['python', 'matlab']
tweets_by_prg_lang = [tweets['python'].value_counts()[True], tweets['matlab'].value_counts()[True]]
print('\nNumber of tweets for each language: ')
print(tweets_by_prg_lang)
x_pos = list(range(len(prg_langs)))
width = 0.4
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width, alpha=1, color='g')
# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: python vs. matlab (Raw data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)

# Targeting relevant tweets
tweets['programming'] = tweets['text'].apply(lambda tweet: word_in_text('programming', tweet))
tweets['tutorial'] = tweets['text'].apply(lambda tweet: word_in_text('tutorial', tweet))
tweets['relevant'] = tweets['text'].apply(lambda tweet: word_in_text('programming', tweet) or word_in_text('tutorial', tweet))
tweets_by_relevant = [tweets['programming'].value_counts()[True],tweets['tutorial'].value_counts()[True],tweets['relevant'].value_counts()[True]]
print('\nNumber of tweets relevant: ')
print(tweets_by_relevant)
print('\nNumber of tweets relevant for each programming language ')
tweets_by_prg_lang = [tweets[tweets['relevant'] == True]['python'].value_counts()[True], 
                      tweets[tweets['relevant'] == True]['matlab'].value_counts()[True]]
print(tweets_by_prg_lang)

# Make a simple comparison chart
x_pos = list(range(len(prg_langs)))
width = 0.5
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width,alpha=1,color='g')
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: python vs. matlab (Relevant data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)
plt.show()

# Extract links from the relevant tweets
def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''

tweets['link'] = tweets['text'].apply(lambda tweet: extract_link(tweet))
tweets_relevant = tweets[tweets['relevant'] == True]
tweets_relevant_with_link = tweets_relevant[tweets_relevant['link'] != '']

print(tweets_relevant_with_link[tweets_relevant_with_link['python'] == True]['link'])
print(tweets_relevant_with_link[tweets_relevant_with_link['matlab'] == True]['link'])



df_text = tweets.loc[tweets['relevant']==True,'text']
df_text.to_csv('D:\\MyProjects\\02_NLP\\Practice\\DataScienceAnalytics', sep='\t', encoding='utf-8')
tweets.loc[:,['text','relevant']].to_csv('D:\\MyProjects\\02_NLP\\Practice\\tweets.csv', sep='\t', encoding='utf-8')





# Clean up data by removing unnecessary characters and altering the format of words
def clean_text(text):
    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


import numpy as np

text = []
text = np.array(df_text)
df_clean_text = []    
for t in text:
    df_clean_text.append(clean_text(t))


min_line_length = 2
max_line_length = 200

# Filter out the questions that are too short/long
short_lines = []
i = 0
for line in df_clean_text:
    if len(line.split()) >= min_line_length and len(line.split()) <= max_line_length:
        short_lines.append(line)
    i += 1


# Create a dictionary for the frequency of the vocabulary
vocab = {}
for line in short_lines:
    for word in line.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1           


# Remove rare words from the vocabulary.
# We will aim to replace fewer than 5% of words with <UNK>
# You will see this ratio soon.
threshold = 1
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1
print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)


# In case we want to use a different vocabulary sizes for the source and target text, 
# we can set different threshold values.
# Nonetheless, we will create dictionaries to provide a unique integer for each word.
status_vocab_to_int = {}
word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        status_vocab_to_int[word] = word_num
        word_num += 1       


# Create dictionaries to map the unique integers to their respective words.
# i.e. an inverse dictionary for vocab_to_int.
status_int_to_vocab = {v_i: v for v, v_i in status_vocab_to_int.items()}



# Sort questions and answers by the length of questions.
# This will reduce the amount of padding during training
# Which should speed up training and help to reduce the loss
sorted_status = []
for length in range(1, max_line_length+1):
    for i in enumerate(status_vocab_to_int):
        if len(i[1]) == length:
            sorted_status.append(status_vocab_to_int[i])
print(len(sorted_status))





















