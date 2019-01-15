import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import Embedding

import numpy as np
import pickle
import json

import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.externals import joblib

import operator

nltk.download('wordnet')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


class QuestionClassfier():
    def __init__(self):
        self.class_dict = {}
        self.class_dict_lookup = {}
        self.model = None
        return

    def load_data(self, fname = 'labeledQuestions.txt'):
        fhdl = open(fname)
        labels = []
        questions = []
        for line in fhdl:
            line = line.split()
            labels.append(line[0])
            questions.append(self.preprocess_data(line[1:]))
        fhdl.close()
        label_set = set(labels)
        counter = 0
        for label in label_set:
            self.class_dict[label] = counter
            counter += 1

        fhdl = open('label_dict.json','w')
        fhdl.write(json.dumps(self.class_dict))

        return questions, labels

    def preprocess_data(self, text):
        processed_text = []
        for word in text:
            '''if word in stopwords:
                print word
                continue'''
            if word.isalpha():
                word = self.lemmatize(word)
                processed_text.append(word)
        return processed_text

    def lemmatize(self, word):
        lemma = lemmatizer.lemmatize(word,'v')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word,'n')
        return lemma

    def getBOW(self, text):
        BOW = {}
        for word in text:
            BOW[word] = BOW.get(word, 0) + 1
            
        return BOW


    def retrieve_vsm(self, vectorizer, questions, label, feature_extractor, train = True, predict = False):
        que_vsm = []
        classification = []
        index = 0
        while index < len(questions):
            que_vsm.append(feature_extractor(questions[index]))
            if not predict:
                classification.append(self.class_dict[label[index]])
            index += 1
        if train:
            dataset = vectorizer.fit_transform(que_vsm)
        else:
            dataset = vectorizer.transform(que_vsm)
        print len(que_vsm), len(classification)

        return dataset, classification

    def build_model(self, voc_size):
        max_features = voc_size
        batch_size = 32
        embedding_dims = 100
        dropout_rate = 0.2
        hidden_dims = 64
        class_size = len(self.class_dict)


        model = Sequential()
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length = max_features))
        model.add(Dropout(dropout_rate))

        model.add(Dense(hidden_dims, activation = 'relu', input_dim = embedding_dims))
        model.add(Dropout(dropout_rate))

        model.add(Dense(hidden_dims, activation = 'relu'))
        model.add(Dropout(dropout_rate))

        model.add(Flatten())

        model.add(Dense(class_size, activation = 'softmax'))

        sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

        model.compile(loss = 'categorical_crossentropy',
                      optimizer = sgd,
                      metrics = ['accuracy'])

        return model

    def train_model(self):
        questions, label = self.load_data()
        train_que = questions[:5000]
        train_label = label[:5000]

        test_que = questions[4500:]
        test_label = label[4500:]

        vectorizer = DictVectorizer()
        train_data, train_label_no = self.retrieve_vsm(vectorizer, train_que, train_label, self.getBOW)
        train_label_vector = keras.utils.to_categorical(train_label_no, num_classes = len(self.class_dict))

        test_data, test_label_no = self.retrieve_vsm(vectorizer, test_que, test_label, self.getBOW, train=False)
        test_label_vector = keras.utils.to_categorical(test_label_no, num_classes = len(self.class_dict))
        voc_size = len(vectorizer.get_feature_names())
        model = self.build_model(voc_size)
        model.fit(train_data, train_label_vector, epochs = 10, batch_size = 100)

        model.save('my_model.h5')
        with open('vectoriser.pk', 'wb') as fhdl:
            pickle.dump(vectorizer, fhdl)

        score = model.evaluate(test_data, test_label_vector, batch_size = 100)
        print score

    def naive_bayes(self):
        questions, label = self.load_data()
        train_que = questions[:4500]
        train_label = label[:4500]

        test_que = questions[4500:]
        test_label = label[4500:]

        vectorizer = DictVectorizer()
        train_data, train_label_no = self.retrieve_vsm(vectorizer, train_que, train_label, self.getBOW)

        test_data, test_label_no = self.retrieve_vsm(vectorizer, test_que, test_label, self.getBOW, train=False)


        clf = MultinomialNB(alpha=0.9)
        clf.fit(train_data, train_label_no)
        prediction = clf.predict(test_data)
        self.checkResults(prediction,test_label_no)



    def checkResults(self, predictions, classifications):
        print "accuracy"
        print accuracy_score(classifications, predictions)
        print classification_report(classifications, predictions)

    def load_nn_model(self, model_fname = 'my_model.h5', vect_fname = 'vectoriser.pk', class_fname = 'label_dict.json'):
        self.model = keras.models.load_model(model_fname)
        self.vectorizer = joblib.load(vect_fname) 
        self.class_dict = json.load(open(class_fname))

        for key in self.class_dict.keys():
            self.class_dict_lookup[self.class_dict[key]] = key


    def question_classify_nn(self, question):
        vectorised_question, discard = self.retrieve_vsm(self.vectorizer, [question], [], self.getBOW, train = False, predict = True)
        predict = self.model.predict(vectorised_question)
        max_index = np.argmax(predict)
        print max_index, np.max(predict)
        print predict
        return self.class_dict_lookup[max_index]




class QuestionClassifierRule():
    def __init__(self):
        self.nn_classifier = QuestionClassfier()
        self.nn_classifier.load_nn_model()
        self.rules_BK = [
            ('who', 'PERSON'),
            ('whom', 'PERSON'),
            ('location', 'LOCATION'),
            ('place', 'LOCATION'),
            ('where', 'LOCATION'),
            ('region', 'LOCATION'),
            ('state', 'LOCATION'),
            (' city', 'LOCATION'),
            ('country', 'LOCATION'),
            ('when', 'NUMBER'),
            ('what time', 'NUMBER'),
            ('much', 'NUMBER'),
            ('many', 'NUMBER'),
            ('percent', 'NUMBER'),
            ('rank', 'NUMBER'),
            ('percentage','NUMBER'),
            ('profit', 'NUMBER'),
            ('number', 'NUMBER'),
            ('date', 'NUMBER'),
            ('year', 'NUMBER'),
            ('population','NUMBER'),
            ('temperature', 'NUMBER')
        ]
        self.rules = [
            ('who', 'PERSON'),
            ('with whom', 'PERSON'),
            ('location ', 'LOCATION'),
            ('what place', 'LOCATION'),
            ('where', 'LOCATION'),
            ('state ', 'LOCATION'),
            (' city ', 'LOCATION'),
            (' country ', 'LOCATION'),
            ('when', 'DATE'),
            ('what time', 'DATE'),
            ('how much', 'NUMBER'),
            ('how many', 'NUMBER'),
            ('percent ', 'NUMBER'),
            ('rank', 'NUMBER'),
            ('percentage','NUMBER'),
            ('ratio', 'NUMBER'),
            ('cost', 'NUMBER'),
            ('price ', 'NUMBER'),
            ('number', 'NUMBER'),
            ('date', 'DATE'),
            ('what year', 'YEAR'),
            ('population ','NUMBER'),
            ('temperature ', 'NUMBER')
        ]

        self.rules_firstBK = [
            ('Who', 'PERSON'),
            ('How much', 'NUMBER'),
            ('How many', 'NUMBER'),
            ('When', 'NUMBER'),

        ]

        self.rules_first = [
            ('Who', 'PERSON'),
            ('How much', 'NUMBER'),
            ('How many', 'NUMBER'),
            ('When', 'DATE'),

        ]



    def classify(self, lemmatised_question,original_que):
        for (token, tag) in self.rules_first:
            if token in original_que:
                return tag
        for (token, tag) in self.rules:
            if token in lemmatised_question:
                return tag
        predicted_tag = self.nn_classifier.question_classify_nn(original_que)
        
        if predicted_tag in ['MONEY', 'PERCENT']:#, 'DATE', 'YEAR']:
            return 'NUMBER'
        elif predicted_tag == 'REASON':
            return 'OTHER'
        else:
            return predicted_tag
        



'''
if __name__ == '__main__':
    qc = QuestionClassfier()
    #qc.train_model()
    #qc.naive_bayes()
    qc.load_nn_model()
    fhdl = open('labeledQuestions.txt')
    counter = 0
    for line in fhdl:
        tag = line.split()[0]
        que = line.split()[1]
        pre_tag = qc.question_classify(que)
        if pre_tag == tag:
            counter += 1
    print counter
    #print qc.question_classify('What films featured the character Popeye Doyle ?')
'''






        






