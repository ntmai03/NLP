"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Sentiment Analysis - Movie review
https://towardsdatascience.com/scikit-learn-for-text-analysis-of-amazon-fine-food-reviews-ea3b232c2c1b
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np

# Import and view data set
df = pd.read_csv('D:\\MyProjects\\02_NLP\\Data\\Reviews.csv')
df.head()
df.dropna(inplace=True)
df[df['Score'] != 3]
df['Positivity'] = np.where(df['Score'] > 3,1,0)
df.head()

# split data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Text'],df['Positivity'],random_state=0)
print('X_train first entry: \n\n', X_train[0])
print('\n\n X_train shape: ',X_train.shape)

# create bag of words from the training set
# CountVectorizer: convert a collection of text documents into a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(X_train)
# get some of the vocabularies by using the get_feature_names method
vect.get_feature_names()[::2000]
len(vect.get_feature_names())

# transform the documents in X_train to a document term matrix, which gives us the bags-of-word representation of X_train
X_train_vectorized = vect.transform(X_train)
X_train_vectorized.toarray()

# Tf–idf term weighting
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(X_train)
len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
# Prediction
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Smallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))
print(model.predict(vect.transform(['The candy is not good, I will never buy them again','The candy is not bad, I will buy them again'])))

# Use ngram model
vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
len(vect.get_feature_names())
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))
print(model.predict(vect.transform(['The candy is not good, I would never buy them again','The candy is not bad, I will buy them again'])))



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Multinomial Naive Bayes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# df1
df = pd.read_csv('D:\\MyProjects\\02_NLP\\Data\\tweets.csv')
df
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text'].values.astype('U').tolist()
count_vect = CountVectorizer()
count_vect.fit(text)
transformed = count_vect.transform(["I love my iphone!!!"])
counts = count_vect.transform(text)
# print(counts.A, counts.toarray())
nb = MultinomialNB()
nb.fit(counts, target)
print(nb.predict(transformed))

# df2
df = pd.read_csv('D:\\MyProjects\\02_NLP\\Data\\Reviews.csv')
df.head()
df.dropna(inplace=True)
df[df['Score'] != 3]
df['Positivity'] = np.where(df['Score'] > 3,1,0)
df.head()
# split data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Text'],df['Positivity'],random_state=0)
count_vect = CountVectorizer()
count_vect.fit(X_train)
transformed = count_vect.transform(["I love my iphone!!!"])
counts = count_vect.transform(transformed)
nb = MultinomialNB()
nb.fit(counts, df['Positivity'])
print(nb.predict(transformed))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Bernoulli Naive Bayes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd

# train data
d1 = [1,1,1,0,0,0,0,0,0]
d2 = [1,1,0,1,1,0,0,0,0]
d3 = [0,1,0,0,1,1,0,0,0]
d4 = [0,1,0,0,0,0,1,1,1]

train_data = np.array([d1,d2,d3,d4])
label = np.array(['B','B','B','N'])

# test data
d5 = [1,0,0,1,0,0,0,1,0]
d6 = [0,1,0,1,1,0,0,1,1]
d9 = [0,1,0,0,0,0,0,1,1]
d6 = np.array(d6).reshape(1,-1)
d9 = np.array(d9).reshape(1,-1)

## Call BernoulliNB
clf = BernoulliNB()
# training
clf.fit(train_data,label)

# Predict
print('Predicting class of d6 in each class:',clf.predict(d9))
print('Probability of d6 in each class:',str(clf.predict_proba(d6)))
test = np.array([d5,d6,d9])
clf.predict_proba(test)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Text manipulation/ Text pre-processing
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
tf.__version__

# Load the data
lines = open('D:\\MyProjects\\02_NLP\\ChatBot\\cornell movie-dialogs corpus\\movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('D:\\MyProjects\\02_NLP\\ChatBot\\cornell movie-dialogs corpus\\movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# The sentences that we will be using to train our model
lines[:10]
# The sentences'ids, which will be processed to become our input and target data
conv_lines[:10]

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
id2line

# Create a list of all of the conversation's lines'ids
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))    
convs[:10]

# Sort the sentences into questions (inputs) and answer (targets)
questions = []
answers = []
for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
questions[0:10]
answers[0:10]
convs[0:10]
       
# Check if we have loaded the data correctly
limit = 0
for i in range(limit, limit+5):
    print(questions[i])
    print(answers[i])
    print()

# Compare lengths of questions and answers
len(questions)
len(answers)

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


# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))


# Take a look at some of the data to ensure that it has been cleaned well.
limit = 0
for i in range(limit, limit+5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()
    

# Find the length of sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])

# basic statistics for length
lengths.describe()
print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))


# Remove questions and answers that are shorter than 2 words and longer than 20 words.
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []
i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out the answers that are too short/long
short_questions = []
short_answers = []
i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

# Compare the number of lines we will use with the total number of lines.
print("# of questions:", len(short_questions))
print("# of answers:", len(short_answers))
print("% of data used: {}%".format(round(len(short_questions)/len(questions),4)*100))

# Create a dictionary for the frequency of the vocabulary
vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1           
for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1


# Remove rare words from the vocabulary.
# We will aim to replace fewer than 5% of words with <UNK>
# You will see this ratio soon.
threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1
print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)


# In case we want to use a different vocabulary sizes for the source and target text, 
# we can set different threshold values.
# Nonetheless, we will create dictionaries to provide a unique integer for each word.
questions_vocab_to_int = {}
word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1       
answers_vocab_to_int = {}
word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1


# Add the unique tokens to the vocabulary dictionaries.
codes = ['<PAD>','<EOS>','<UNK>','<GO>']
for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int)+1   
for code in codes:
    answers_vocab_to_int[code] = len(answers_vocab_to_int)+1
   
    
# Create dictionaries to map the unique integers to their respective words.
# i.e. an inverse dictionary for vocab_to_int.
questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}


# Check the length of the dictionaries.
print(len(questions_vocab_to_int))
print(len(questions_int_to_vocab))
print(len(answers_vocab_to_int))
print(len(answers_int_to_vocab))


# Add the end of sentence token to the end of every answer.
for i in range(len(short_answers)):
    short_answers[i] += ' <EOS>'


# Convert the text to integers. 
# Replace any words that are not in the respective vocabulary with <UNK> 
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)   
answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answers_int.append(ints)
# Check the lengths
print(len(questions_int))
print(len(answers_int))


# Calculate what percentage of all words have been replaced with <UNK>
word_count = 0
unk_count = 0
for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1   
for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1    
unk_ratio = round(unk_count/word_count,4)*100    
print("Total number of words:", word_count)
print("Number of times <UNK> is used:", unk_count)
print("Percent of words that are <UNK>: {}%".format(round(unk_ratio,3)))


# Sort questions and answers by the length of questions.
# This will reduce the amount of padding during training
# Which should speed up training and help to reduce the loss
sorted_questions = []
sorted_answers = []
for length in range(1, max_line_length+1):
    for i in enumerate(questions_int):
        if len(i[1]) == length:
            sorted_questions.append(questions_int[i[0]])
            sorted_answers.append(answers_int[i[0]])
print(len(sorted_questions))
print(len(sorted_answers))
print()
for i in range(3):
    print(sorted_questions[i])
    print(sorted_answers[i])
    print()



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


# create 2 charts:
# 1st: the top 5 languages, 2nd: the top countries
tweets_by_lang = tweets['lang'].value_counts()
print(tweets_by_lang)
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')


tweets_by_country = tweets['country'].value_counts()
print(tweets_by_country)
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Countries', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 countries', fontsize=15, fontweight='bold')
tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')
 plt.show()

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






































