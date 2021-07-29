import re
import jieba
from gensim.corpora import Dictionary
class Base:
    text_corpus_dict = {}
    text_dictionary_dict = {}
    text_dict = {}
    @classmethod
    def build_dictionary(cls,text_path,use_stop_word):
        cls.text_dict[text_path]= cls.preprocess_text(text_path,use_stop_word)
        cls.text_dictionary_dict[text_path] = Dictionary(cls.text_dict[text_path])
        cls.text_corpus_dict[text_path] = [ cls.text_dictionary_dict[text_path].doc2bow( doc) for doc in cls.text_dict[text_path]]
    @classmethod
    def preprocess_text(cls,text_path,use_stop_word):
        with open(text_path,mode='r',encoding='utf-8') as f:
             text = [line.strip() for line in f.readlines()]
             return [list(filter(lambda x:len(x)!=0,map(lambda x:re.sub('[^\u4e00-\u9fa5]',"",x),jieba.cut(doc))))  for doc in text if len(doc)>0]






