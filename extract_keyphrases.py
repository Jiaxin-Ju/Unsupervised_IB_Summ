"""
Description:
"""

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# key phrases extraction using rake
class KeyphrasesExtraction:
    # initialization
    def __init__(self):
        # initialize here, thus do not need to repeat again and again
        self.stop_words = stopwords.words("english")
        self.punc = string.punctuation
        self.lemma = WordNetLemmatizer()

    # function preprocess the text
    def preprocess(self, doc):
        doc = doc.lower()  # lowercasing all letters
        doc = re.sub(r"\s+", " ", doc)  # remove redundant spaces
        doc = nltk.word_tokenize(doc)  # tokenize the text
        return doc

    # extract phrases by stop words and puncuations
    def rake_phrases(self, doc):
        phrases = []
        phrase = []
        # loop for each word in the doc
        for i in range(len(doc)):
            # if the word is stop word or word is punctuation
            if doc[i] in self.stop_words or doc[i] in self.punc:
                # if the phrases contains something, add it to overall phrase
                if phrase and len(phrase) > 1:
                    phrases.append(phrase)
                # reset phrase
                phrase = []
            else:
                # keep adding word
                phrase.append(self.lemma.lemmatize(doc[i]))

        # do a check at the end
        if phrase:
            phrases.append(phrase)
        return phrases

    # function calculate the score for each word
    def calculate_words_score(self, phrases):
        # calculate the degree and frequency for each word
        freq = {}
        degree = {}

        # go through each phrase
        for phrase in phrases:
            # go through each word in the phrase
            for word in phrase:
                # update frequency and degree
                if word not in freq:
                    freq[word] = 1
                    degree[word] = len(phrase)
                else:
                    freq[word] += 1
                    degree[word] += len(phrase)

        # calculate score for each word, by experiment using the degree only is the best
        words_score = {}
        for word in freq:
            words_score[word] = degree[word]
        return words_score

    # function to calculate the score for phrases
    def calculate_phrases_score(self, phrases, words_score):
        phrases_score = {}
        # convert phrase into string
        for i in range(len(phrases)):
            phrases[i] = " ".join(phrases[i])

        # get unique phrases
        phrases = list(set(phrases))
        for phrase in phrases:
            temp_score = 0
            # go calculate score, set phrase back to string first
            phrase_list = phrase.split(" ")
            # add up score of words in the phrase
            for word in phrase_list:
                temp_score += words_score[word]

            # unlike the original approach, take length into consideration as well
            phrases_score[phrase] = temp_score / len(phrase_list)
        return phrases_score

    # function get the top 10 key phrases
    def get_topN_phrases(self, n, phrases_score):
        # sort dictionary
        top_n = sorted(phrases_score.items(), key=lambda item: item[1], reverse=True)[:n]
        selected = []

        # get the string only
        for pair in top_n:
            selected.append(pair[0])
        return selected

    # function process the text and return key phrases
    def process(self, n, doc):
        doc = self.preprocess(doc)  # preprocess the doc
        phrases = self.rake_phrases(doc)  # extract phrases
        words_score = self.calculate_words_score(phrases)  # calculate score for words
        phrases_score = self.calculate_phrases_score(phrases, words_score)  # calculate score for phrases
        key_phrases = self.get_topN_phrases(n, phrases_score)  # get top N key words
        return key_phrases


if __name__ == "__main__":
    f = open('scientific_paper_test/arxiv/arxiv_test_1.txt', 'r')
    file = f.readlines()
    f.close()
    text = file[0]
    rake = KeyphrasesExtraction()
    extracted_key_phrases = rake.process(10, text)
    print(extracted_key_phrases)
