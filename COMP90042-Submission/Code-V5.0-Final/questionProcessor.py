import json
import textProcessor
import nltk
import re


nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

nltk.download('maxent_ne_chunker')

nltk.download('wordnet')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

['CD','FW','JJ','JJR','JJS','NN','NNS','NNP','NNPS','RB','VB','VBD','VBG','VBN','VNP','VBZ']


class QuestionProcessor (textProcessor.TextProcessor): 
    def __init__(self, question):
        self.question = question
        self.query = []
        self.term_list = ['NN','NNP','NNS']
        self.term_list2 = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','CD','FW']
        self.term_list3 = ['RB','VB','VBD','VBG','VBN','VNP','VBZ']
        
        


    def query_extract2(self):

        sent = self.keywords_extract2()

        full_query = self.keywords_extract_full()
        #sent = self.pos_tagger(self.question)
        
        for token in sent:
            #potential improvement to consider unify last name and full name to one term
            if self.check_stopwords(token):
                continue
            self.query.append(token)

        print "selfquery", self.query

        return self.query, full_query


    def keywords_extract2(self):
        keywords = []

        #tagged_sent = self.sf_pos_tagger(self.question.strip('?'))
        tagged_sent = self.pos_tagger(self.question.strip('?'))

        sent_dict = {}
        for (token, tag) in tagged_sent:
            sent_dict[token] = tag


        sent = self.question.split()
        for token in sent:
            tag = sent_dict.get(token, None)
            if tag:
                if tag in self.term_list2:
                    token = self.preprocess(token)
                    keywords.append(token)
                else:
                    continue
            else:
                token = self.preprocess(token)
                keywords.append(token)

        return set(keywords)

    def keywords_extract_full(self):
        full_result = []
        for token in self.question.split():
            token = self.preprocess(token)
            if self.check_stopwords(token):
                continue
            else:
                full_result.append(token)
        return set(full_result)




