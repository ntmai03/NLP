############################################################################
# Bag-of-words
############################################################################
# Load qdap
install.packages('qdap')
install.packages('SnowballC')
install.packages('wordcloud')
install.packages('tm')
install.packages('viridisLite')
install.packages('dendextend')
install.packages('ggplot2')
install.packages('RWeka')
library(qdap)
library(SnowballC)
library(wordcloud)
library(tm)
library(viridisLite)
library(plyr)
library(dendextend)
library(ggplot2)
library(RWeka)
library(stringr)


# Print new_text to the console
new_text <- "NLP is a way for computers to analyze, understand, and derive meaning from human language in a smart and useful way. By utilizing NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation"

# Find the 10 most frequent terms: term_count
term_count <- freq_terms(new_text, 10)

# Plot term_count
plot(term_count)

############################################################################
# Coffee Dataset
############################################################################
# Load corpus 
coffee_tweets <- read.csv("C:\\DataScience\\DataSet\\coffee.csv", stringsAsFactors = FALSE) 
str(coffee_tweets)

# Isolate text from tweets: coffee_tweets
coffee_tweets <- coffee_tweets$text

# View first 5 tweets 
head(coffee_tweets, 5)

# Make a vector source: coffee_source 
coffee_source <- VectorSource(coffee_tweets) 
head(coffee_source, 5)

# Make a volatile corpus: coffee_corpus 
coffee_corpus <- VCorpus(coffee_source) 

# Print the 15th tweet in coffee_corpus
coffee_corpus[[15]]

# Print the contents of the 15th tweet in coffee_corpus
coffee_corpus[[15]][1]

# Now use content to review plain text
content(coffee_corpus[[10]])

# Stem words 
stem_words <- stemDocument(c("complicatedly", "complicated", "complication")) 
stem_words

# Complete words using single word dictionary 
stemCompletion(stem_words, c("complicate"))

# Complete words using entire corpus
# stemCompletion(stem_words, coffee_corpus)

# List standard English stop words
stopwords("en")

# Print text without standard stop words
removeWords(coffee_corpus, stopwords("en"))

# Add "coffee" and "bean" to the list: new_stops
new_stops <- c("coffee", "bean", stopwords("en"))

# Remove stop words from text
removeWords(coffee_corpus, new_stops)

# Apply various preprocessing functions
tm_map(coffee_corpus, removeNumbers) 
tm_map(coffee_corpus, removePunctuation) 
tm_map(coffee_corpus, content_transformer(replace_abbreviation))


# Generate TDM 
coffee_tdm <- TermDocumentMatrix(coffee_corpus) 
# Convert coffee_dtm to a matrix: coffee_m
coffee_m <- as.matrix(coffee_dtm)
# Print the dimensions of coffee_m
dim(coffee_m)
# Review a portion of the matrix
coffee_m[475:480, 2593:2598]

# Generate DTM 
coffee_dtm <- DocumentTermMatrix(coffee_corpus)
# Print coffee_tdm data
coffee_tdm
# Convert coffee_tdm to a matrix: coffee_m
coffee_m <- as.matrix(coffee_tdm)

# Print the dimensions of the matrix
dim(coffee_m)

# Review a portion of the matrix
coffee_m[2593:2598, 475:480]

# Create frequency
frequency <- freq_terms(
  coffee_tweets, 
  top = 10, 
  at.least = 3, 
  stopwords = "Top200Words"
)

# Make a frequency barchart
plot(frequency)

terms_vec <- frequency$WORD
term_frequency <- frequency$FREQ
wordcloud(terms_vec, term_frequency, max.words = 50, colors = "red")


############################################################################
# Word clouds
############################################################################
# Load corpus 
chardonnay_tweets <- read.csv("C:\\DataScience\\DataSet\\chardonnay.csv", stringsAsFactors = FALSE) 
str(chardonnay_tweets)

# Find the 10 most frequent terms: term_count
frequency <- freq_terms(chardonnay_tweets$text)

# Vector of terms
terms_vec <- names(term_frequency)
terms_vec <- frequency$WORD
term_frequency <- frequency$FREQ
wordcloud(terms_vec, term_frequency, max.words = 200, colors = "red")

# Isolate text from tweets: coffee_tweets
chardonnay_tweets <- chardonnay_tweets$text

# View first 5 tweets 
head(chardonnay_tweets, 5)

# Make a vector source: coffee_source 
chardonnay_source <- VectorSource(chardonnay_tweets) 
head(chardonnay_source, 5)

# Make a volatile corpus: coffee_corpus 
chardonnay_corp <- VCorpus(chardonnay_source) 

# Print the 15th tweet in coffee_corpus
chardonnay_corp[[15]][1]


clean_corpus <- 
  function
(corpus){ 
  corpus <- tm_map(corpus, removePunctuation) 
  corpus <- tm_map(corpus, stripWhitespace) 
  corpus <- tm_map(corpus, removeNumbers) 
  corpus <- tm_map(corpus, content_transformer(tolower)) 
  corpus <- tm_map(corpus, removeWords,  
                   c(stopwords(
                     "en"
                   ), 
                   "amp"
                   )) 
  return(corpus)
}
chardonnay_corp <- clean_corpus(chardonnay_corp)

# Review a "cleaned" tweet
content(chardonnay_corp[[24]])

# Add to stopwords
stops <- c(stopwords(kind = 'en'), 'chardonnay')

# Review last 6 stopwords 
tail(stops)

# Apply to a corpus
cleaned_chardonnay_corp <- tm_map(chardonnay_corp, removeWords, stops)

# Review a "cleaned" tweet again
content(cleaned_chardonnay_corp[[24]])

# Generate TDM 
chardonnay_tdm <- TermDocumentMatrix(cleaned_chardonnay_corp) 
# Convert coffee_dtm to a matrix: coffee_m
chardonnay_m <- as.matrix(chardonnay_tdm)
# Review a portion of the matrix
chardonnay_m[100:500, 1:8]
chardonnay_words = rowSums(chardonnay_m)
head(chardonnay_words)

# Sort the chardonnay_words in descending order
sorted_chardonnay_words <- sort(chardonnay_words, decreasing = TRUE)

# Print the 6 most frequent chardonnay terms
head(sorted_chardonnay_words)

# Get a terms vector
terms_vec <- names(chardonnay_words)

# Create a wordcloud for the values in word_freqs
wordcloud(terms_vec, chardonnay_words, max.words = 50, colors = "red")

# Print the list of colors
colors()

# Print the wordcloud with the specified colors
wordcloud(terms_vec, chardonnay_words, max.words = 100, 
          colors = c("grey80", "darkgoldenrod1", "tomato"))


# Select 5 colors 
color_pal <- cividis(n = 5)

# Examine the palette output
color_pal 

# Create a wordcloud with the selected palette
wordcloud(terms_vec, chardonnay_words, 
          max.words = 100, colors = color_pal)




# Find common words across multiple documents
# Create all_coffee
all_coffee <- paste(coffee_tweets, collapse = " ")

# Create all_chardonnay
all_chardonnay <- paste(chardonnay_tweets, collapse = " ")

# Create all_tweets
all_tweets <- c(all_coffee, all_chardonnay)

# Convert to a vector source
all_tweets <- VectorSource(all_tweets)

# Create all_corpus
all_corpus <- VCorpus(all_tweets)

# Clean the corpus
all_clean <- clean_corpus(all_corpus)

# Create all_tdm
all_tdm <- TermDocumentMatrix(all_clean)

# Create all_m
all_m <- as.matrix(all_tdm)

# Print a commonality cloud
commonality.cloud(all_m, max.words = 100, colors = "steelblue1")



# Visualize dissimilar words

# Clean the corpus
all_clean <- clean_corpus(all_corpus)

# Create all_tdm
all_tdm <- TermDocumentMatrix(all_clean)

# Give the columns distinct names
colnames(all_tdm) <- c("coffee", "chardonnay")

# Create all_m
all_m <- as.matrix(all_tdm)
head(all_m)

# Create comparison cloud
comparison.cloud(all_m, colors = c("orange", "blue"), max.words = 50)


# Polarized tag cloud
top25_df <- all_m %>%
  # Convert to data frame
  as.data.frame(rownames = "word") %>% 
  # Keep rows where word appears everywhere
  filter(all.vars(. > 0)) %>% 
  # Get difference in counts
  mutate(all_m %>% as.data.frame(rownames = "word"),difference = chardonnay - coffee) %>% 
  # Keep rows with biggest difference
  top_n(25, wt = difference) %>% 
  # Arrange by descending difference
  arrange(desc(difference))



# Word association
word_associate(coffee_tweets, match.string = "barista", 
               stopwords = c(Top200Words, "coffee", "amp"), 
               network.plot = TRUE, cloud.colors = c("gray85", "darkred"))
# Add title
title(main = "Barista Coffee Tweet Associations")


############################################################################
# EX3
############################################################################
# Print the dimensions of tweets_tdm
dim(all_tdm)

# Create tdm1
tdm1 <- removeSparseTerms(coffee_tdm, sparse = 0.95)

# Create tdm2
tdm2 <- removeSparseTerms(coffee_tdm, sparse = 0.975)

# Print tdm1
tdm1

# Print tdm2
tdm2


# Create tweets_tdm2
tweets_tdm2 <- removeSparseTerms(chardonnay_tdm, sparse = 0.975)

# Create tdm_m
tdm_m <- as.matrix(tweets_tdm2)

# Create tweets_dist
tweets_dist <- dist(tdm_m)

# Create hc
hc <- hclust(tweets_dist)

# Plot the dendrogram
plot(hc)



# Create hcd
hcd <- as.dendrogram(hc)

# Print the labels in hcd
labels(hcd)

# Change the branch color to red for "marvin" and "gaye"
hcd_colored <- branches_attr_by_labels(hcd, c("marvin", "gaye"), "red")

# Plot hcd
plot(hcd_colored, main = "Better Dendrogram")

# Add cluster rectangles 
rect.dendrogram(hcd_colored, k = 2, border = "grey50")





# Create associations
associations <- findAssocs(tweets_tdm2, "marvin", 0.0)

# View the venti associations
associations

# Create associations_df
associations_df <- list_vect2df(associations, col2 = "word", col3 = "score")

# Plot the associations_df values
ggplot(associations_df, aes(score, word)) + 
  geom_point(size = 3) 



# Make tokenizer function 
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}

# Create unigram_dtm
unigram_dtm <- DocumentTermMatrix(cleaned_chardonnay_corp)

# Create bigram_dtm
bigram_dtm <- DocumentTermMatrix(
  cleaned_chardonnay_corp, 
  control = list(tokenize = tokenizer)
)

# Print unigram_dtm
unigram_dtm

# Print bigram_dtm
bigram_dtm


# Create bigram_dtm_m
bigram_dtm_m <- as.matrix(bigram_dtm)

# Create freq
freq <- colSums(bigram_dtm_m)

# Create bi_words
bi_words <- names(freq)

# Examine part of bi_words
str_subset(bi_words, "^marvin")


# Plot a wordcloud
wordcloud(bi_words, freq, max.words = 15)

# Create a TDM
tdm <- TermDocumentMatrix(coffee_corpus)

# Convert it to a matrix
tdm_m <- as.matrix(tdm)

# Examine part of the matrix
tdm_m[c("coffee", "espresso", "latte"), 161:166]

# Make a clean volatile corpus: text_corpus
text_corpus <- clean_corpus(VCorpus(coffee_source))

# Examine the first doc content
content(text_corpus[[1]])

# Access the first doc metadata
meta(text_corpus[1])



############################################################################
# EX4: Let's learn something about how employees review both Amazon and Google
############################################################################
# Load corpus 
amzn <- read.csv("C:\\DataScience\\DataSet\\500_amzn.csv", stringsAsFactors = FALSE) 
goog <- read.csv("C:\\DataScience\\DataSet\\500_goog.csv", stringsAsFactors = FALSE) 


# Print the structure of amzn
str(amzn)

# Create amzn_pros
amzn_pros <- amzn$pros

# Create amzn_cons
amzn_cons <- amzn$cons

# Print the structure of goog
str(goog)

# Create goog_pros
goog_pros <- goog$pros

# Create goog_cons
goog_cons <- goog$cons

# qdap_clean the text
qdap_cleaned_amzn_pros <- qdap_clean(amzn_pros)
qdap_cleaned_amzn_pros <- amzn_pros

# Source and create the corpus
amzn_p_corp <- VCorpus(VectorSource(qdap_cleaned_amzn_pros))

# tm_clean the corpus
amzn_pros_corp <- clean_corpus(amzn_p_corp)


# qdap_clean the text
qdap_cleaned_goog_pros <- qdap_clean(goog_pros)
qdap_cleaned_goog_pros <- goog_pros

# Source and create the corpus
goog_p_corp <- VCorpus(VectorSource(qdap_cleaned_goog_pros))

# tm_clean the corpus
goog_pros_corp <- clean_corpus(goog_p_corp)


# Create amzn_p_tdm
amzn_p_tdm <- TermDocumentMatrix(
  amzn_pros_corp, 
  # control = list(tokenize = tokenizer)
)

qdap_cleaned_amzn_cons <- amzn_cons

# Source and create the corpus
amzn_c_corp <- VCorpus(VectorSource(qdap_cleaned_amzn_cons))

# tm_clean the corpus
amzn_cons_corp <- clean_corpus(amzn_c_corp)


# Source and create the corpus
goog_p_corp <- VCorpus(VectorSource(qdap_cleaned_goog_pros))

# tm_clean the corpus
goog_pros_corp <- clean_corpus(goog_p_corp)


qdap_cleaned_goog_cons <- goog_cons

# Source and create the corpus
goog_c_corp <- VCorpus(VectorSource(qdap_cleaned_goog_cons))

# tm_clean the corpus
goog_cons_corp <- clean_corpus(goog_c_corp)


# Create amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(
  amzn_cons_corp, 
 # control = list(tokenize = tokenizer)
)

# Create amzn_c_tdm_m
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)

# Create amzn_c_freq
amzn_c_freq <- rowSums(amzn_c_tdm_m)


# Create amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(
  amzn_cons_corp,
  # control = list(tokenize = tokenizer)
)

# Print amzn_c_tdm to the console
amzn_c_tdm

# Create amzn_c_tdm2 by removing sparse terms 
amzn_c_tdm2 <- removeSparseTerms(amzn_c_tdm, .993)

# Create hc as a cluster of distance values
hc <- hclust(dist(amzn_c_tdm2), 
             method = "complete")

# Produce a plot of hc
plot(hc)


# Create amzn_p_tdm
amzn_p_tdm <- TermDocumentMatrix(
  amzn_pros_corp, 
  # control = list(tokenize = tokenizer)
)

# Create amzn_p_m
amzn_p_m <- as.matrix(amzn_p_tdm)

# Create amzn_p_freq
amzn_p_freq <- rowSums(amzn_p_m)





# Create all_goog_corp
all_goog_corp <- tm_clean(all_goog_corpus)

# Create all_tdm
all_tdm <- TermDocumentMatrix(all_goog_corp)

# Create all_m
all_m <- as.matrix(all_tdm)

# Build a comparison cloud
comparison.cloud(all_m, 
                 colors = c("#F44336", "#2196f3"), 
                 max.words = 100)


# Filter to words in common and create an absolute diff column
common_words <- all_tdm_df %>% 
  filter(
    AmazonPro != 0,
    GooglePro != 0
  ) %>%
  mutate(diff = abs(AmazonPro - GooglePro))

# Extract top 5 common bigrams
(top5_df <- top_n(common_words, 5, diff))

# Create the pyramid plot
pyramid.plot(top5_df$AmazonPro, top5_df$GooglePro, 
             labels = top5_df$terms, gap = 12, 
             top.labels = c("Amzn", "Pro Words", "Goog"), 
             main = "Words in Common", unit = NULL)


# Extract top 5 common bigrams
(top5_df <- top_n(common_words, 5, diff))

# Create a pyramid plot
pyramid.plot(
  # Amazon on the left
  top5_df$AmazonNeg, 
  # Google on the right
  top5_df$GoogleNeg, 
  # Use terms for labels
  labels = top5_df$terms, 
  # Set the gap to 12
  gap = 12, 
  # Set top.labels to "Amzn", "Neg Words" & "Goog"
  top.labels = c("Amzn", "Neg Words", "Goog"), 
  main = "Words in Common", 
  unit = NULL
)








