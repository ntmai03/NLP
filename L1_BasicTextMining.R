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
library(ggplot2)
library(RWeka)
library(stringr)

# Print new_text to the console
quotes_text <- read.csv("D:\\MyProjects\\02_NLP\\Data\\Quotes.csv", stringsAsFactors = FALSE) 
str(quotes_text)
# Find the 10 most frequent terms: term_count
term_count <- freq_terms(quotes_text$text, 20)
# Plot term_count
plot(term_count)

# Create frequency
term_count <- freq_terms(
  quotes_text$text, 
  top =25, 
  at.least = 6, 
  stopwords = "Top200Words"
)
# Make a frequency barchart
plot(term_count)
terms_vec <- term_count$WORD
term_frequency <- term_count$FREQ
wordcloud(terms_vec, term_frequency,max.words = 25, colors = "red")

# Make a vector source: source 
source <- VectorSource(quotes_text$text) 
head(source, 5)
# Make a volatile corpus: coffee_corpus 
corpus <- VCorpus(source) 
# Print the 15th tweet in coffee_corpus
corpus[[1]]
# Print the contents of the 15th tweet in coffee_corpus
corpus[[1]][1]
# Now use content to review plain text
content(corpus[[1]])

# Stem words 
stem_words <- stemDocument(c("complicatedly", "complicated", "complication")) 
print(stem_words)

# List standard English stop words
stopwords("en")
# Print text without standard stop words
removeWords(corpus[[1]], stopwords("en"))
# Add "NLP" and "language" to the list: new_stops
new_stops <- c("the", "you", stopwords("en"))
# Remove stop words from text
removeWords(corpus[[1]], new_stops)

# Apply various preprocessing functions
clean_corpus <- function(corpus){ 
  corpus <- tm_map(corpus, removePunctuation) 
  corpus <- tm_map(corpus, content_transformer(replace_abbreviation))
  corpus <- tm_map(corpus, stripWhitespace) 
  corpus <- tm_map(corpus, removeNumbers) 
  corpus <- tm_map(corpus, content_transformer(tolower))
  # Add to stopwords
  stops <- c(stopwords(kind = 'en'), 'us','can','will')
  # remove stop words
  corpus <- tm_map(corpus, removeWords,  stops) 
  corpus <- tm_map(corpus, stripWhitespace) 

  return(corpus)
}
content(corpus[[1]])
corpus <- clean_corpus(corpus)
# Review a "cleaned" tweet
content(corpus[[1]])

# Generate TDM 
tdm <- TermDocumentMatrix(corpus) 
# Convert tdm to a matrix: tdm_m
tdm_m <- as.matrix(tdm)
# Print the dimensions of coffee_m
dim(tdm_m)
# Review a portion of the matrix
tdm_m[1:15, 1:10]

quote_words = rowSums(tdm_m)
head(quote_words)
# Sort the chardonnay_words in descending order
sorted_quote_words <- sort(quote_words, decreasing = TRUE)
# Print the 6 most frequent chardonnay terms
head(sorted_quote_words)
# Get a terms vector
terms_vec <- names(sorted_quote_words)
# Create a wordcloud for the values in word_freqs
wordcloud(terms_vec, sorted_quote_words, max.words = 50, colors = "red")
# Print the list of colors
colors()
# Print the wordcloud with the specified colors
wordcloud(terms_vec, sorted_quote_words, max.words = 50, colors = c("grey80", "darkgoldenrod1", "tomato"))
# Select 5 colors 
color_pal <- cividis(n = 5)
# Examine the palette output
color_pal 
# Create a wordcloud with the selected palette
wordcloud(terms_vec, sorted_quote_words, max.words = 50, colors = color_pal)


############################################################################
# EX3
############################################################################
# Print the dimensions of tdm
dim(tdm)
# Create tdm1
tdm1 <- removeSparseTerms(tdm, sparse = 0.95)
dim(tdm1)
# Create tdm2
tdm2 <- removeSparseTerms(tdm, sparse = 0.975)
dim(tdm2)
# Create tdm_m
tdm_m <- as.matrix(tdm2)
tdm_m[1:11, 1:10]

library(ggplot2)
library(RWeka)
library(stringr)

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
associations <- findAssocs(tdm2, "marvin", 0.0)
# View the venti associations
associations
# Create associations_df
associations_df <- list_vect2df(associations, col2 = "word", col3 = "score")
# Plot the associations_df values
ggplot(associations_df, aes(score, word)) + geom_point(size = 3) 



# Make tokenizer function 
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}

# Create unigram_dtm
unigram_dtm <- DocumentTermMatrix(corpus)

# Create bigram_dtm
bigram_dtm <- DocumentTermMatrix(
  corpus, 
  control = list(tokenize = tokenizer)
)

# Print unigram_dtm
unigram_dtm

# Print bigram_dtm
bigram_dtm


# Create bigram_dtm_m
bigram_dtm_m <- as.matrix(bigram_dtm)
bigram_dtm_m[1:10,1:10]

# Create freq
freq <- colSums(bigram_dtm_m)

# Create bi_words
bi_words <- names(freq)

# Examine part of bi_words
str_subset(bi_words, "^marvin")


# Plot a wordcloud
wordcloud(bi_words, freq, max.words = 15)

# Create a TDM
tdm <- TermDocumentMatrix(corpus)

# Convert it to a matrix
tdm_m <- as.matrix(tdm)





