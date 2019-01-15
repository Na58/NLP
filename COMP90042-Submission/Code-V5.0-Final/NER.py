from nltk.tag import StanfordNERTagger
import re

class NERTagger():
    def __init__(self):
        stanford_ner_dir = '/Users/Rena/StandfordParserData/stanford-ner-2018-02-27/'
        eng_model_filename= stanford_ner_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'
        my_path_to_jar= stanford_ner_dir + 'stanford-ner.jar'

        self.tagger = StanfordNERTagger(model_filename=eng_model_filename, path_to_jar=my_path_to_jar)
        self.ner_cache = {}
        self.time_list = [
            'january',
            'february',
            'march',
            'april',
            'may',
            'june',
            'july',
            'august',
            'september',
            'october',
            'november',
            'december'
        ]
        self.ordinal_list = [
            'first',
            'largest',
            'highest',
            'second',
            'third',
            'fourth',
            'fifth',
            'one',
            'two',
            'three',
            'four',
            'five',
            'six',
            'seven',
            'eight',
            'nine',
            'ten'
        ]

    def cache_sents(self, sents):
        ###cache the documents###
        tokenised_sent = map(lambda x: x.split(), sents)
        tagged = self.tagger.tag_sents(tokenised_sent)
        for i in range(len(tokenised_sent)):
            self.ner_cache[sents[i]] = tagged[i]
        return True


    def tag(self, sents):

        pattern = '([^A-Z]\.\s[A-Z])'
        if re.search(pattern, sents):
            sentences = self.split_para(sents)
            entity_list = []
            for s in sentences:
                
                try:
                    tagged = self.ner_cache[s]
                except KeyError:
                    sent = map(self.strip_word, s.split())
                    tagged = self.tagger.tag(sent)
                    self.ner_cache[s] = tagged
                entity = self.entity_parse(tagged)
                entity_list.append(entity)
            result = sum(entity_list, [])
        else:
            
            try:
                tagged_sents = self.ner_cache[sents]
            except KeyError:
                sen = map(self.strip_word, sents.split())
                tagged_sents = self.tagger.tag(sen)
                self.ner_cache[sents] = tagged_sents

            result = self.entity_parse(tagged_sents)
        return result

    def strip_word(self, word):
        pattern = ('"",:.?!;''')
        return word.strip(pattern)


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




    def entity_parse_detail(self, tagged_sent):
        ###entity parsing method for detailed tagset###
        start = True
        retagged_entity = []
        for item in tagged_sent:
            token, tag = item
            if token.lower() in self.time_list:
                tag = 'MONTH'
            elif re.match('^[1|2][0-9]{3,3}$', token):
                tag = 'YEAR'
            elif token.lower() in self.ordinal_list:
                tag = 'NUMBER'
            elif tag == 'ORGANIZATION':
                tag = 'OTHER'
            elif tag == 'O':
                if not start and len(token) > 0 and token[0].isupper():
                    tag = 'OTHER'
                elif any(char == '%' for char in token):
                    tag = 'NUMBER'#'PERCENT'
                elif any(char == '$' for char in token):
                    tag = 'NUMBER'#'MONEY'
                elif any(char.isdigit() for char in token):
                    tag = 'NUMBER'

            if start:
                start = False
            retagged_entity.append((token, tag))

        retagged_entity = self.retag_date(retagged_entity)

        return self.gather_entity(retagged_entity)

    def retag_date(self, tagged_entity):
        ###gather NUMBER MONTH YEAR pattern into DATE###
        result_entity = []
        i = 0
        while i < len(tagged_entity) - 2:
            (token1, tag1) = tagged_entity[i]

            if tag1 == 'NUMBER':
                (token2, tag2) = tagged_entity[i+1]
                if tag2 == 'MONTH':
                    (token3, tag3) = tagged_entity[i+2]
                    if tag3 == 'YEAR':
                        result_entity.append((token1,'DATE'))
                        result_entity.append((token2, 'DATE'))
                        result_entity.append((token3, 'DATE'))
                        i = i + 3
                        continue
            elif tag1 == 'MONTH':
                (token2, tag2) = tagged_entity[i+1]
                if tag2 == 'NUMBER':
                    (token3, tag3) = tagged_entity[i+2]
                    if tag3 == 'YEAR':
                        result_entity.append((token1,'DATE'))
                        result_entity.append((token2, 'DATE'))
                        result_entity.append((token3, 'DATE'))
                        i = i + 3
                        continue
            result_entity.append((token1, tag1))
            i += 1
        for counter in range(len(tagged_entity) - i):
            result_entity.append(tagged_entity[i + counter])

        return result_entity
            


    def entity_parse(self, tagged_sent):
        ###entity parsing for general tagset###
        start = True
        retagged_entity = []
        for item in tagged_sent:
            token, tag = item
            if token.lower() in self.time_list:
                tag = 'NUMBER'
            if token.lower() in self.ordinal_list:
                tag = 'NUMBER'
            if tag == 'ORGANIZATION':
                tag = 'OTHER'
            if tag == 'O':
                if not start and len(token) > 0 and token[0].isupper():
                    tag = 'OTHER'
                elif any(char.isdigit() for char in token):
                    tag = 'NUMBER'

            if start:
                start = False
            retagged_entity.append((token, tag))



        return self.gather_entity(retagged_entity)

    def gather_entity(self, retagged_entity):
        ###gather continuous entities###
        gathered_entity = []
        tag = 'O'
        token = ''
        for (new_token, new_tag) in retagged_entity:
            if tag == new_tag:
                
                token = token + ' ' + new_token

            else:
                if tag != 'O':
                    
                    gathered_entity.append((token, tag))
                tag = new_tag
                token = new_token
        if tag != 'O':
            
            gathered_entity.append((token, tag))

        return gathered_entity


'''
if __name__ == '__main__':
    ner = NERTagger()
    ner.tag('In 1925, the first meeting of the Hundred Percent Club, composed of IBM salesmen who meet their quotas, convened in Atlantic City, New Jersey.')
'''








