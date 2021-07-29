from gensim.models import TfidfModel
from base import Base
import jieba
import itertools
class TFIDF(Base):
    def __init__(self,file_path,use_stop_word = True):
        self.file_path= file_path
        if (super().__getattribute__('text_corpus_dict').get(file_path,0)  == 0):
             super().build_dictionary(file_path,use_stop_word)
        self.build_model()
    def extract_keyword(self,number = 100):
        self.tfidfvalue = [self.model[tokens] for tokens in super().__getattribute__('text_corpus_dict').get(self.file_path)]
        tfidf_sorted = sorted(itertools.chain(*self.tfidfvalue),key = lambda x:x[1],reverse = 1)
        number = number if number<=len(tfidf_sorted) else len(tfidf_sorted)
        tmp_dict = super().__getattribute__('text_dictionary_dict').get(self.file_path)
        key_words = [ tmp_dict[item[0]] for item in  tfidf_sorted ]
        return key_words[:number]
    def build_model(self):
        self.model = TfidfModel(super().__getattribute__('text_corpus_dict').get(self.file_path))

