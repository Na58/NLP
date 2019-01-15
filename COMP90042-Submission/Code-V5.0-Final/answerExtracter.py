import nltk
nltk.download('wordnet')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
import re

import questionClassifier, textProcessor
from nltk.tag import StanfordPOSTagger, StanfordNERTagger


class AnswerExtracter():
    def __init__(self):
        stanford_pos_dir = '/Users/Rena/StandfordParserData/stanford-postagger-full-2018-02-27/'
        eng_model_filename= stanford_pos_dir + 'models/english-bidirectional-distsim.tagger'
        my_path_to_jar= stanford_pos_dir + 'stanford-postagger.jar'
        self.sf_pos_tagger = StanfordPOSTagger(model_filename=eng_model_filename, path_to_jar=my_path_to_jar)

        self.text_processor = textProcessor.TextProcessor()
        self.rule_classifier = questionClassifier.QuestionClassifierRule()
        self.closed_class_tag = [
            'CC',
            'DT',
            'IN',
            'MD',
            'PRP',
            'PRP$',
            'WDT',
            'WP',
            'WP$'
        ]

    def extract_answer(self, entity_list, question, sent):
        result = self.extract_entity(entity_list, question, sent)

        if result:
            answer, tag = result
            answer = self.normalise(answer)
            return answer
        return ''

    def normalise(self, text):
        text = text.replace('"', ' ').replace('(',' ( ').replace(')',' ) ').replace('\'s',' \'s').replace(';', ' ; ').replace(':', ' : ').replace('s\'', 's \'')
        normalised = ''
        for token in text.split():
            token = token.lower().strip('.,()[]""')
            if token.strip('%') != token:
                normalised += token.strip('%') + ' %'
            elif token.strip('$') != token:
                normalised += '$ ' + token.strip('$')
            elif token.strip(u"\xA3") != token:
                normalised += u"\xA3 " + token.strip(u"\xA3")
            else:
                normalised += token + ' '
        return normalised.strip(' ')


    def extract_entity(self, entity_list, question, sent):
        pri_entity = self.prioritise_content(entity_list, question)
        type_entity = self.prioritise_type(pri_entity, question)
        print "###eliminate different type###", type_entity
        for entities in type_entity:
            for entity in entities:
                if len(entity) == 0:
                    continue
                elif len(entity) == 1:
                    return entity[0]
                else:
                    ranked = self.rank_proximity(entity, question, sent, closed = False)

                    if ranked and len(ranked) == 1:
                        return ranked[0]
                    elif ranked and len(ranked) > 1:
                        print "multiple entity, checking closed word"
                        close_ranked = self.rank_proximity(ranked, question, sent)
                        if close_ranked:
                            return close_ranked[0]
                    else:
                        print "no entity, checking closed word"
                        close_ranked = self.rank_proximity(entity, question, sent)
                        if close_ranked:
                            return close_ranked[0]
                    return entity[0]
        return None
        
    def prioritise_content(self, entity_list, question):
        ###Group the entity into 2###
        ###1. High_priority - Not in question###
        ###2. Low_priority - In Question###
        high_priority = []
        low_priority = []
        lemmatised_question = map(self.lemmatise, question.split())
        for (token, tag) in entity_list:
            if self.check_exist(token, lemmatised_question):
                low_priority.append((token, tag))
            else:
                high_priority.append((token, tag))

        return [high_priority, low_priority]

    def check_exist(self, token, lemmatised_question):
        ###only return true if every part in the entity exist in question###
        words = token.split()
        words = map(self.lemmatise, words)
        for word in words:
            if word in lemmatised_question:
                continue
            else:
                return False
        return True


    def prioritise_type(self, prioritised_entity, question):
        ###group entity according to correct and incorrect type###
        expected_tag = self.get_answer_type(question)
        print 'expected tag', expected_tag
        type_prioritised = []
        for entity in prioritised_entity:
            corr_type = []
            wron_type = []
            for (token, tag) in entity:
                if tag == expected_tag:
                    corr_type.append((token, tag))
                else:
                    wron_type.append((token, tag))
            type_prioritised.append([corr_type, wron_type])
        return type_prioritised


    def rank_proximity(self, entity_list, question, sent, closed = True):
        ###distance to open/closed class in the original sentence###
        normalised_sent = map(self.text_processor.preprocess, sent.split())
        normalised_sent = ' '.join(normalised_sent).strip(' ')
        que_closed_word = self.filter_closed(self.text_processor.pos_tagger(question), closed)
        if len(que_closed_word) != 0:
            result = []
            for entity in entity_list:
                dist = self.calculate_dist(entity, que_closed_word, normalised_sent)
                token, tag = entity
                result.append((dist, (token,tag)))
            result = sorted(result, key = lambda x:x[0])
            if len(result) == 1:
                return [result[0][1]]
            else:
                top_ent = self.extract_same_rank(result)
                return [entity_tuple for (dist, entity_tuple) in top_ent]
        return None

    def extract_same_rank(self, ranked_dist_ent):
        ###extract entity with same rank for further voting###
        top_ent = []
        best_dist, best_ent = ranked_dist_ent[0]
        for i in range(len(ranked_dist_ent)):
            dist, ent = ranked_dist_ent[i]
            if dist > best_dist:
                break
            else:
                top_ent.append((dist, ent))
        return top_ent

    def calculate_dist(self, entity, que_closed_word, normalised_sent):
        token, tag = entity
        token = ' '.join(map(self.lemmatise, token.split())).strip(' ')
        splitted = normalised_sent.split(token)
        
        if len(splitted) == 2:
            left_sent = splitted[0].split()
            right_sent = splitted[1].split()
            print left_sent, '\n', right_sent
            l_in = 1000
            r_in = 1000
            for i in range(len(left_sent)-1, 0, -1):
                if left_sent[i] in que_closed_word:
                    l_in = i
                    break
            for i in range(len(right_sent)):
                if right_sent[i] in que_closed_word:
                    r_in = i
                    break
            return min(l_in, r_in)
        if len(splitted) == 1:
            if re.search(normalised_sent[0], token):
                new_sent = re.sub(token, '', normalised_sent)
                new_sent = new_sent.split()
                r_in = 1000
                if new_sent:
                    for i in range(len(new_sent)):
                        if new_sent[i] in que_closed_word:
                            r_in = i
                            break
                return r_in
            elif re.search(normalised_sent[-1], token):
                new_sent = re.sub(token, '', normalised_sent)
                new_sent = new_sent.split()
                l_in = 1000
                if new_sent:
                    for i in range(len(new_sent)-1, 0, -1):
                        if new_sent[i] in que_closed_word:
                            l_in = i
                            break
                return l_in
        else:
            return 1000



    def sf_pos(self, sent):
        pattern = '^[A-Za-z0-9]+$'
        tokens = [t for t in sent.split() if re.match(pattern, t)]
        return self.sf_pos_tagger.tag(tokens)


    def filter_closed(self, tagged_sent, closed = True):
        filtered_word = []
        for (token, tag) in tagged_sent:
            if tag in self.closed_class_tag:
                if closed:
                    try:
                        filtered_word.append(self.lemmatise(token))
                    except:
                        filtered_word.append(token)
            else:
                if not closed:
                    try:
                        filtered_word.append(self.lemmatise(token))
                    except:
                        filtered_word.append(token)
        return filtered_word




    def lemmatise(self, word):
        word = word.strip('"",:.?!;''()')
        lemma = lemmatizer.lemmatize(word,'v')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word,'n')
        return lemma.lower()

    def get_answer_type(self, question):
        lemmatised_question = map(self.lemmatise, question.split())

        return self.rule_classifier.classify(lemmatised_question,question)



'''
if __name__ == '__main__':
    ae = AnswerExtracter()
    que = 'On what date did Universal City Studios open?'
    lemma_que = map(ae.lemmatise, que.split())
    que_l = ' '.join(lemma_que)
    print ae.get_answer_type(que)
    entity = [(u'1925,', 'NUMBER'), (u'Hundred Percent Club,', 'OTHER'), (u'IBM', u'OTHER'), (u'Atlantic', u'LOCATION'), (u'City, New Jersey.', 'OTHER')]
    print ae.extract_answer(entity, 'when was the Hundred Percent Club first meeting?')
'''











