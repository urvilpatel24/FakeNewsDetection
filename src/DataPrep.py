# -*- coding: utf-8 -*-

#import os
import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb


class DataPrep(object):
 
    def __init__(self, train_filename, test_filename, valid_filename):
        self.train_news = pd.read_csv(train_filename)
        self.test_news = pd.read_csv(test_filename)
        self.valid_news = pd.read_csv(valid_filename)
        self.eng_stemmer = SnowballStemmer('english')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.porter = PorterStemmer()
        nltk.download('treebank')
        nltk.download('stopwords')
       
 
    #data observation
    def data_obs(self):
        print("training dataset size:")
        print(self.train_news.shape)
        print(self.train_news.head(10))

        #below dataset were used for testing and validation purposes
        print(self.test_news.shape)
        print(self.test_news.head(10))

        print(self.valid_news.shape)
        print(self.valid_news.head(10))

    #check the data by calling below function
    #data_obs()

    #distribution of classes for prediction
    def create_distribution(self, dataFile):
        return sb.countplot(x='Label', data=dataFile, palette='hls')

    #data integrity check (missing label values)
    #none of the datasets contains missing values therefore no cleaning required
    def data_qualityCheck(self):

        print("Checking data qualitites...")
        self.train_news.isnull().sum()
        self.train_news.info()

        print("check finished.")

        #below datasets were used to 
        self.test_news.isnull().sum()
        self.test_news.info()

        self.valid_news.isnull().sum()
        self.valid_news.info()

    #run the below function call to see the quality check results
    #data_qualityCheck()


    #Stemming
    def stem_tokens(self, tokens, stemmer):
        stemmed = []
        for token in tokens:
            stemmed.append(stemmer.stem(token))
        return stemmed

    #process the data
    def process_data(self, data, exclude_stopword=True, stem=True):
        tokens = [w.lower() for w in data]
        tokens_stemmed = tokens
        tokens_stemmed = self.stem_tokens(tokens, self.eng_stemmer)
        tokens_stemmed = [w for w in tokens_stemmed if w not in self.stopwords ]
        return tokens_stemmed


    #creating ngrams
    #unigram 
    def create_unigram(self, words):
        assert type(words) == list
        return words

    #bigram
    def create_bigrams(self, words):
        assert type(words) == list
        skip = 0
        join_str = " "
        Len = len(words)
        if Len > 1:
            lst = []
            for i in range(Len-1):
                for k in range(1,skip+2):
                    if i+k < Len:
                        lst.append(join_str.join([words[i],words[i+k]]))
        else:
            #set it as unigram
            lst = self.create_unigram(words)
        return lst

    """
    #trigrams
    def create_trigrams(words):
        assert type(words) == list
        skip == 0
        join_str = " "
        Len = len(words)
        if L > 2:
            lst = []
            for i in range(1,skip+2):
                for k1 in range(1, skip+2):
                    for k2 in range(1,skip+2):
                        for i+k1 < Len and i+k1+k2 < Len:
                            lst.append(join_str.join([words[i], words[i+k1],words[i+k1+k2])])
            else:
                #set is as bigram
                lst = create_bigram(words)
        return lst
    """


   

    def tokenizer(self, text):
        return text.split()


    def tokenizer_porter(self, text):
        return [self.porter.stem(word) for word in text.split()]

    #doc = ['runners like running and thus they run','this is a test for tokens']
    #tokenizer([word for line in test_news.iloc[:,1] for word in line.lower().split()])

    #show the distribution of labels in the train and test data
    """def create_datafile(filename)
        #function to slice the dataframe to keep variables necessary to be used for classification
        return "return df to be used"
    """

    """#converting multiclass labels present in our datasets to binary class labels
    for i , row in data_TrainNews.iterrows():
        if (data_TrainNews.iloc[:,0] == "mostly-true" | data_TrainNews.iloc[:,0] == "half-true" | data_TrainNews.iloc[:,0] == "true"):
            data_TrainNews.iloc[:,0] = "true"
        else :
            data_TrainNews.iloc[:,0] = "false"

    for i,row in data_TrainNews.iterrows():
        print(row)
    """

