import re
import jieba
from gensim.corpora import Dictionary
class Base:
    text_path = ''
    corpus = ''
    @classmethod
    def build_dictionary(cls,text_path,use_stop_word):
        cls.text_path = text_path
        cls.text = cls.preprocess_text(text_path,use_stop_word)
        cls.dic = Dictionary(cls.text)
        cls.corpus = [cls.dic.doc2bow( doc) for doc in cls.text]
    @classmethod
    def preprocess_text(cls,text_path,use_stop_word):
        with open(text_path,mode='r',encoding='utf-8') as f:
             text = [line.strip() for line in f.readlines()]
             return [list(filter(lambda x:len(x)!=0,map(lambda x:re.sub('[^\u4e00-\u9fa5]',"",x),jieba.cut(doc))))  for doc in text if len(doc)>0]






