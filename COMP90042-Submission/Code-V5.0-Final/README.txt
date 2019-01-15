answerExtracter.py
Extract answer from the input entity list, based on
-- Entity Type
-- Question Content
-- Distance to Open/Closed Class in Sentence

NER.py
Tag the named entity in the input sentence

paragraphSearch.py
search for the best matching paragraph and sentence

questionAnswer.py
Invoking method

questionClassifier.py
contain three models:
-- Neural Network
-- Naive Bayes
-- Rule Based Classifer

textProcessor.py
basic class for text processing

questionProcessor.py
normalise the questions

vectoriser.pk
vectoriser used in model training

my_model.h5
neural network model

label_dict.json
number-label mapping for neural network model

csvFromatter.py
format the output to csv



documentSearch.py
base class for paragraphSearch.py

