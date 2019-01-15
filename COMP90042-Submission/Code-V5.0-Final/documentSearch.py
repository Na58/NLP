import json
import math

class DocSearcher ():
    def __init__(self, invert_file, doc_Len_file):
        self.inverted_list = json.load(open(invert_file))
        self.doc_length = json.load(open(doc_Len_file))
        self.total_doc = len(self.doc_length)
        self.lavg = float(sum(self.doc_length.values()))/self.total_doc
    

    def search(self, query):
        ###only retrieve doc that contains all query terms
        term_freq = {}
        term_doc_freq = {}
        shared_doc = self.doc_length.keys()

        for term in query:
            try:
                term_doc_freq[term] = self.inverted_list[term]
                term_freq[term] = len(term_doc_freq[term])
                shared_doc = self.intersection(shared_doc, self.inverted_list[term].keys())
            except:
                continue
    
        scored_doc = self.tf_idf(term_freq, term_doc_freq, shared_doc)

        return scored_doc

    def search_treshold(self, query):
        term_freq = {}
        term_doc_freq = {}
        shared_doc = []

        for term in query:
            try:
                tdf = self.inverted_list[term]
                tf = len(tdf)

                if tf >= 100:
                    continue
                else:
                    term_freq[term] = tf
                    term_doc_freq[term] = tdf
                    map(shared_doc.append, tdf.keys())
            except:
                continue
        if len(shared_doc) == 0:
            print "NO VALID QUERY"
        scored_doc = self.tf_idf(term_freq, term_doc_freq, shared_doc)

        return scored_doc



    def tf_idf(self, term_freq, term_doc_freq, shared_doc):
        scored_doc = []

        query_terms = term_freq.keys()

        for docid in shared_doc:
            score = 0
            for term in query_terms:
                score += self.bm25(term_freq[term], term_doc_freq[term].get(docid, 0), self.doc_length[docid], 
                                    self.lavg, self.total_doc)
                #score += float(term_freq[term]) * (float(self.total_doc) / term_doc_freq[term].get(docid, 0))

            #score = score/self.doc_length[docid]
            scored_doc.append((docid, score))
        return scored_doc

    def bm25(self, ft, fdt, ld, lavg, N):
        # k1 = 1.75 b = 0.75
        tf = math.log((N - ft + 0.5)/(ft + 0.5))
        idf = 2.7 * fdt / (1.7 * (0.25 + 0.75 * ld / lavg) + fdt )

        return (tf * idf / ld)


    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3



#DocSearcher('invertedIndex.json', 'docLength.json')            

