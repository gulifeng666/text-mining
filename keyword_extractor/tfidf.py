from gensim.models import TfidfModel,LdaModel
from base import Base
import jieba
import itertools
from functools import reduce
class LDA(Base):
    def __init__(self,file_path,use_stop_word = True,stop_save_file = None):
        self.file_path= file_path
        self.key_words = []
        if (super().__getattribute__('text_corpus_dict').get(file_path,0)  == 0):
             super().build_dictionary(file_path,use_stop_word,stop_save_file)
        self.build_model()
    def extract_keyword(self,save_file = None,topicnum = 10,number = 100):
        key_words_dict = [[{item[0]:item[1]} for item in self.model.show_topic(i,number//topicnum)] for i in range(topicnum)]
        key_words_dict= list(itertools.chain(*key_words_dict))
        def dict_add(x, y):
            x[list(y.keys())[0]] = x.get(list(y.keys())[0], 0) + list(y.values())[0]
            return x
        key_words = sorted(reduce(lambda x, y: dict_add(x, y), key_words_dict).items(), key=lambda x: x[1], reverse=True)
        key_words = [item[0] for item in key_words]
        number = number if number <= len(key_words) else len(key_words)
        if (save_file != None):
            self.to_file(key_words[:number], save_file)
        self.key_words = key_words[:number]
        return key_words[:number]
    def build_model(self):
        self.model = LdaModel(super().__getattribute__('text_corpus_dict').get(self.file_path),num_topics=10,id2word=super().__getattribute__('text_dictionary_dict').get(self.file_path))

class TFIDF(Base):
    def __init__(self,file_path,use_stop_word = True,stop_save_file = None):
        self.file_path= file_path
        self.key_words = []
        if (super().__getattribute__('text_corpus_dict').get(file_path,0)  == 0):
             super().build_dictionary(file_path,use_stop_word,stop_save_file)
        self.build_model()
    def extract_keyword(self,save_file = None,number = 100):
        self.tfidfvalue = [self.model[tokens] for tokens in super().__getattribute__('text_corpus_dict').get(self.file_path)]
        tfidf_sorted = sorted(itertools.chain(*self.tfidfvalue),key = lambda x:x[1],reverse = 1)
        tmp_dict = super().__getattribute__('text_dictionary_dict').get(self.file_path)
        key_words = [ tmp_dict[item[0]] for item in  tfidf_sorted]
        key_words_dict=[{tmp_dict[item[0]]:item[1] }for item in tfidf_sorted]
        def dict_add(x,y):
            x[list(y.keys())[0]]=x.get(list(y.keys())[0],0)+list(y.values())[0]
            return x
        key_words = sorted(reduce( lambda x,y: dict_add(x,y),key_words_dict).items(),key = lambda x:x[1],reverse=True)
        key_words = [item[0] for item in key_words]
        #key_words = sorted(set(key_words),key = key_words.index)
        number = number if number<=len(key_words) else len(key_words)
        if(save_file!=None):
            self.to_file(key_words[:number],save_file)
        self.key_words = key_words[:number]
        return key_words[:number]

    def build_model(self):
        self.model = TfidfModel(super().__getattribute__('text_corpus_dict').get(self.file_path))

