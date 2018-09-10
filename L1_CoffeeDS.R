############################################################################
# Load corpus 
############################################################################
coffee_tweets <- read.csv("D:\\MyProjects\\02_NLP\\Data\\coffee.csv", stringsAsFactors = FALSE) 
# View the structure of tweets
str(coffee_tweets)
# Use the text column
coffee_tweets <- coffee_tweets$text
# View first 5 tweets 
head(coffee_tweets, 5)

############################################################################
# Examine data 
############################################################################
# Find the 10 most frequent terms: term_count
term_count <- freq_terms(coffee_tweets, 10)
# Make a bar chart of frequent terms
plot(term_count)

############################################################################
# Data Preprocessing
############################################################################
# Make a vector source: coffee_source 
coffee_source <- VectorSource(coffee_tweets) 
head(coffee_source, 5)
# Make a volatile corpus: coffee_corpus 
coffee_corpus <- VCorpus(coffee_source) 
# Print the 15th tweet in coffee_corpus
coffee_corpus[[15]]
# Print the contents of the 15th tweet in coffee_corpus
coffee_corpus[[15]][1]
content(coffee_corpus[[15]])

# Add "coffee" and "bean" to the list: new_stops
new_stops <- c("coffee", "bean", stopwords("en"))
# Remove stop words from text
removeWords(coffee_corpus, new_stops)

# Apply various preprocessing functions
tm_map(coffee_corpus, removeNumbers) 
tm_map(coffee_corpus, removePunctuation) 
tm_map(coffee_corpus, content_transformer(replace_abbreviation))



