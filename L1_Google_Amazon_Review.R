library(qdap)
library(SnowballC)
library(wordcloud)
library(tm)
library(viridisLite)
library(plyr)
library(ggplot2)
library(RWeka)
library(stringr)


# Load corpus 
amzn <- read.csv("D:\\MyProjects\\02_NLP\\Data\\500_amzn.csv", stringsAsFactors = FALSE) 
goog <- read.csv("D:\\MyProjects\\02_NLP\\Data\\500_goog.csv", stringsAsFactors = FALSE) 


# qdap cleaning function
qdap_clean <- function(x)  {
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <-  replace_ordinal(x)
  x <-  replace_symbol(x)
  x <-  tolower(x)
  
  return(x)
}


# tm cleaning function
clean_corpus <- function(corpus) {
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"), "Google", "Amazon", "company"))
  return(corpus)
}


# Remove punctuation
rm_punc <- removePunctuation(text_data)

# Create character vector
n_char_vec <- unlist(strsplit(rm_punc, split = ' '))

# Perform word stemming: stem_doc
stem_doc <- stemDocument(n_char_vec)

# Re-complete stemmed document: complete_doc
complete_doc <- stemCompletion(stem_doc, comp_dict)

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


# Make tokenizer function 
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}

# Create amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(
  amzn_cons_corp, 
  control = list(tokenize = tokenizer)
)

library(ggplot2)
library(RWeka)
library(stringr)


# Create amzn_c_tdm_m
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)

# Create amzn_c_freq
amzn_c_freq <- rowSums(amzn_c_tdm_m)


# Create amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(
  amzn_cons_corp,
  control = list(tokenize = tokenizer)
)

# Create amzn_p_tdm
amzn_p_tdm <- TermDocumentMatrix(
  amzn_pros_corp, 
  control = list(tokenize = tokenizer)
)

# Create amzn_p_m
amzn_p_m <- as.matrix(amzn_p_tdm)

# Create amzn_p_freq
amzn_p_freq <- rowSums(amzn_p_m)


# Sort term_frequency in descending order
amzn_p_freq <- sort(amzn_p_freq, decreasing = TRUE) 
  
# Plot a barchart of the 10 most common words
barplot(amzn_p_freq[1:10], col = "tan", las = 2)


# Plot a wordcloud using amzn_p_freq values
wordcloud(names(amzn_p_freq), amzn_p_freq, max.words = 25, color = "red")

# Create amzn_p_tdm2 by removing sparse terms
amzn_p_tdm2 <- removeSparseTerms(amzn_p_tdm, sparse = .993) > 
  
  # Create hc as a cluster of distance values
hc <- hclust(dist(amzn_p_tdm2, method = "euclidean"), method = "complete") > 
  
  # Produce a plot of hc
plot(hc)


# Find associations with Top 2 most frequent words
findAssocs(amzn_p_tdm, "great benefits", 0.2)
findAssocs(amzn_p_tdm, "good pay", 0.2)

# Create all_goog_corp
all_goog_corp <- tm_clean(all_goog_corpus) > # Create all_tdm
all_tdm <- TermDocumentMatrix(all_goog_corp)



# Name the columns of all_tdm
colnames(all_tdm) <- c("Goog_Pros", "Goog_Cons") > # Create all_m
all_m <- as.matrix(all_tdm) > # Build a comparison cloud
comparison.cloud(all_m, colors = c("#F44336", "#2196f3"), max.words = 100)



# Create common_words
common_words <- subset(all_tdm_m, all_tdm_m[,1] > 0 & all_tdm_m[,2] > 0)
str(common_words)


# Create difference
difference <- abs(common_words[,1]- common_words[,2]) >
  
  # Add difference to common_words
  common_words <- cbind(common_words, difference) > head(common_words)


# Order the data frame from most differences to least
common_words <- common_words[order(common_words[,"difference"],decreasing = TRUE),]

# Create top15_dftop15_df <- data.frame(x = common_words[1:15,1], y = common_words[1:15,2], labels = rownames(common_words[1:15,]))

# Create the pyramid plot
pyramid.plot(top15_df$x, top15_df$y,
               labels = top15_df$labels, gap = 12,
               top.labels = c("Amzn", "Pro Words", "Google"),
               main = "Words in Common", unit = NULL)

