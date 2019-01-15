import json
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

nltk.download('maxent_ne_chunker')

nltk.download('wordnet')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


from nltk.tag import StanfordPOSTagger, StanfordNERTagger

import re



class TextProcessor ():
    def __init__(self):
        pass


    def preprocess(self, word):
        ###lemmatisation & lowercasing###
        word = word.strip('(),.?!;')
        lemma = lemmatizer.lemmatize(word,'v')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word,'n')
        return lemma.lower()

    def check_stopwords(self, word):
        if word in stopwords:
            return True

        return False


    def pos_tagger(self, para):
        tokens = [t.encode('utf-8') for t in para.split() if t.isalpha()]
        tagged_sent = nltk.pos_tag(tokens)
        return tagged_sent



    def sf_pos_tagger(self, para):
        
        stanford_pos_dir = '/Users/Rena/StandfordParserData/stanford-postagger-full-2018-02-27/'
        eng_model_filename= stanford_pos_dir + 'models/english-bidirectional-distsim.tagger'
        my_path_to_jar= stanford_pos_dir + 'stanford-postagger.jar'

        tagger = StanfordPOSTagger(model_filename=eng_model_filename, path_to_jar=my_path_to_jar) 

        pattern = '^[A-Za-z0-9]+$'
        tokens = [t for t in para.split() if re.match(pattern, t)]

        return tagger.tag(tokens)


    def sf_ner_tagger(self, para):
    	
        stanford_ner_dir = '/Users/Rena/StandfordParserData/stanford-ner-2018-02-27/'
        eng_model_filename= stanford_ner_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'
        my_path_to_jar= stanford_ner_dir + 'stanford-ner.jar'

        tagger = StanfordNERTagger(model_filename=eng_model_filename, path_to_jar=my_path_to_jar) 

        pattern = '^[A-Za-z]+$'
        tokens = [t for t in para.split() if re.match(pattern, t)]

        return tagger.tag(tokens)

'''
if __name__ == '__main__':
    print lemmatizer.lemmatize('Computing-Tabulating-Recording')
    exit()

    searcher = DS.DocSearcher('invertedIndex.json', 'docLength.json')

    questions = json.load(open('devel.json'))
    for question in questions:
        q = question['question']
        print q
        que = QuestionProcessor(q)
        que.query_extract2()
        print que.query
        searcher.search(que.query)
        exit()
'''    
   






