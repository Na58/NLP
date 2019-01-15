import questionProcessor, paragraphSearch, questionClassifier
import NER, answerExtracter

import json
import time

class QA():
    def __init__(self):
        self.questions = None
        self.predictions = []
        self.para_searcher = paragraphSearch.ParaSearcher()
        self.sent_searcher = paragraphSearch.SentSearcher()
        self.classifier = questionClassifier.QuestionClassfier()
        self.ner_tagger = NER.NERTagger()
        self.extracter = answerExtracter.AnswerExtracter()
        self.cached_doc = []

#testing.json
#devel.json
    def load_question(self, fname = 'devel.json'):
        self.questions = json.load(open(fname))


    def predict_answer(self, threshold=-1):
        ###testing method based on dev dataset###
        self.load_question()
        total_que = len(self.questions)
        f_score = 0.0
        counter = 0
        for question in self.questions:
            if counter == total_que or counter == threshold:
                break
            counter += 1
            que_sent = question['question']
            docid = question['docid']
            answer = question['text']
            ranked_sent_list = self.retrieve_ranked_index(que_sent, docid)
            predict = self.extract_answer(ranked_sent_list, que_sent)
            f_score += self.get_fscore(predict, answer)
            
            print "###counter", counter
            print  "###question", que_sent
            print "###answer", answer
            print "###prediction", predict
            print "###f_socre", f_score

        print f_score/counter

    def predict_answer_test(self):
        ###generate the prediction result on test data###
        self.load_question(fname = 'testing.json')
        answer_file = open('answers.csv', 'w')
        answer_file.write('id,answer\n')
        for question in self.questions:
            que_sent = question['question']
            docid = question['docid']
            query_id = question['id']
            ranked_sent_list = self.retrieve_ranked_index(que_sent, docid)
            predict = self.extract_answer(ranked_sent_list, que_sent)
            answer_file.write('%s,%s\n' % (str(query_id), predict.encode('utf8')))
        print 'completed'




    def get_fscore(self, predict, answer):
        predict = predict.split()
        answer = answer.split()
        false_neg = 0.0
        true_pos = 0.0
        f_score = 0
        total_attempt = len(predict)
        for token in answer:
            if token in predict:
                true_pos += 1
            else:
                false_neg += 1
        try:
            p = true_pos / total_attempt
        except:
            p = 0
        try:
            r = true_pos / (true_pos + false_neg)
        except:
            r = 0
        try:
            f_score += 2*p*r / (p + r)
        except:
            f_score += 0

        return f_score

    def retrieve_ranked_index(self, question, docid):
        query = questionProcessor.QuestionProcessor(question)
        keywords, full_keywords = query.query_extract2()


        para_list = self.para_searcher.retrieve_para(docid)
        para_list = map(lambda x: x.strip(' '), para_list)
        para = ' '.join(para_list).strip(' ')
        self.cache_doc(docid, para)

        ranked_sent_list = self.sent_searcher.get_sents(para, full_keywords)

        return ranked_sent_list

    def cache_doc(self, docid, para):
        print docid, self.cached_doc
        if docid not in self.cached_doc:
            splitted_para = self.sent_searcher.split_para(para)
            result = self.ner_tagger.cache_sents(splitted_para)
            if result:
                self.cached_doc.append(docid)
        else:
            print "###Doc cache already exist", self.cache_doc


    def extract_answer(self, ranked_sent_list, question):
        answer = ''
        for i in range(len(ranked_sent_list)):
            entity_list= self.ner_tagger.tag(ranked_sent_list[i])
            if entity_list:
                answer = self.extracter.extract_answer(entity_list, question, ranked_sent_list[i])
                if len(answer) != 0:
                    return answer
        return answer


if __name__ == '__main__':
    startTime = time.time()
    qa = QA()
    qa.predict_answer_test()
    #qa.predict_answer()
    time_used = "---Total Time---\n" + str(time.time() - startTime)
    print time_used





