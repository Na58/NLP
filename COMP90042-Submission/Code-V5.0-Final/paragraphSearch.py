import json
import documentSearch
import textProcessor
import re

import nltk
nltk.download('wordnet')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

stemmer = nltk.stem.PorterStemmer()

class ParaSearcher(documentSearch.DocSearcher):
    def __init__(self, fname = 'documents.json'):
        self.docs = json.load(open(fname))
        self.doc_length = None
        self.total_doc = None
        self.lavg = None

    def get_para(self, docid, keywords):
        para_list = self.retrieve_para(docid)
        
        keywords = map(self.preprocess, keywords)
        ranked_para_index = self.search(para_list, keywords)
        if len(ranked_para_index) != 0:
            return para_list[ranked_para_index[0][0]]#, ranked_para_index[0][0]

        return para_list[0]


    def search(self, para_list, keywords, unique = False):
        para_num = len(para_list)
        term_para_freq = self.get_para_freq(para_list, keywords)
        query_num = len(term_para_freq)

        term_freq = {}

        shared_para = []

        for term in keywords:
            tdf = term_para_freq.get(term, dict())
            tf = len(tdf)
            '''
            if tf == 1 and unique:
                return [(tdf.keys()[0], 1)]
            '''
            if tf > para_num/2 and query_num > 1:
                continue
            term_freq[term] = tf
            term_para_freq[term] = tdf
            map(shared_para.append, tdf.keys())

        if len(shared_para) == 0:
            print "NO VALID QUERY"
        scored_sent = self.tf_idf(term_freq, term_para_freq, shared_para)

        sorted_score_sent = sorted(scored_sent, key = lambda x: x[1], reverse = True)

        #if len(sorted_score_sent) != 0:
        #    return para_list[sorted_score_sent[0][0]],sorted_score_sent[0][0]
        
        return sorted_score_sent


    def retrieve_para(self, docid):
        para = self.docs[int(docid)]['text']
        #para_list = re.split('[.|!|?]\s',para)

        return para

    def get_para_freq(self, para_list, keywords):
        sent_counter = 0
        term_sent_freq = {}
        self.doc_length = dict()
        for sent in para_list:
            self.doc_length[sent_counter] = len(sent)
            tokenised_sent = sent.split()
            temp_dict = {}
            for token in tokenised_sent:
                token = self.preprocess(token)
                if token and token in keywords:
                    temp_dict[token] = temp_dict.get(token, 0) + 1
            #
            for k in temp_dict.keys():
                temp = term_sent_freq.setdefault(k, dict())
                temp[sent_counter] = temp.setdefault(sent_counter, 0) + temp_dict[k]

            sent_counter += 1
        self.total_doc = len(self.doc_length)
        self.lavg = float(sum(self.doc_length.values()))/self.total_doc

        return term_sent_freq




    def preprocess(self, word):
        word = word.strip('()[]{};,""'':.!?')
        lemma = lemmatizer.lemmatize(word,'v')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word,'n')

        if lemma in stopwords:
            return None
        
        return stemmer.stem(lemma.lower())


class SentSearcher(ParaSearcher):
    def __init__(self):
        pass

    def get_sents(self, para, keywords):
        keywords = map(self.preprocess, keywords)
        sent_list = self.split_para(para)
        ranked_sent_index = self.search(sent_list, keywords, unique = True)
        ranked_index = [index for (index, score) in ranked_sent_index]
        '''
        if len(ranked_index) > 0:
            result = []
            for index in ranked_index[:5]:
                result.append(sent_list[index])

            return ' '.join(result).strip(' ')
        return para
        
        if len(ranked_index) != 0:
            best_index = ranked_index[0]
            
            if best_index < len(sent_list) - 1:
                result = sent_list[best_index] + ' ' + sent_list[best_index+1]
            else:
                result = sent_list[best_index]
            
            result = sent_list[best_index]
            return result
        '''
        ranked_sent_list = [sent_list[index] for index in ranked_index]

        return ranked_sent_list

    def split_para(self, para):
        pattern = '([^A-Z]\.\s[A-Z])'
        splitted = re.split(pattern, para)
        matching_pattern = '^[^A-Z]\.\s[A-Z]$'

        for i in range(len(splitted)):
            if re.match(matching_pattern, splitted[i]):
                symbols = splitted[i].split()
                try:
                    splitted[i-1] += symbols[0]
                    splitted[i+1] = symbols[1] + splitted[i+1]
                except:
                    continue

        proper_splitted = []
        for i in range(len(splitted)):
            if re.match(matching_pattern, splitted[i]):
                continue
            else:
                proper_splitted.append(splitted[i])

        return proper_splitted

'''
if __name__ == '__main__':
    searcher = ParaSearcher()
    que = 'On what date did the companies that became the Computing-Tabulating-Recording Company get consolidated?'
    print map(searcher.preprocess, que.split())
'''












