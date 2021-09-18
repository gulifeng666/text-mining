import re
import jieba
from gensim.corpora import Dictionary
import itertools
from tqdm import tqdm
import os
class Base:
    text_corpus_dict = {}
    text_dictionary_dict = {}
    text_dict = {}
    @classmethod
    def build_dictionary(cls,text_path,use_stop_word,stop_save_file):
        cls.text_dict[text_path]= cls.preprocess_text(text_path,use_stop_word,stop_save_file)
        cls.text_dictionary_dict[text_path] = Dictionary(cls.text_dict[text_path])
        cls.text_corpus_dict[text_path] = [ cls.text_dictionary_dict[text_path].doc2bow( doc) for doc in cls.text_dict[text_path]]
    @classmethod
    def preprocess_text(cls,text_path,use_stop_word=False,stop_save_file = None):
        if(use_stop_word):
           if(stop_save_file==None):
                raise FileNotFoundError
           with open(stop_save_file,mode='r',encoding='utf-8') as f:
               cls.stopwords = {word.strip():1 for word in f.readlines()}
        with open(text_path,mode='r',encoding='utf-8') as f:
             text = f.readlines()
             text = ''.join(itertools.chain(*text)).split('\n\n')
             text = map(lambda x:x[1:] if x[0]=='\n' else x,text)
             text = [' '.join(filter(lambda x: '@' not in x,t.split(' '))) for t in text]
             return list(filter(lambda x:len(x)!=0,[list(filter(lambda x:x!='' and cls.stopwords.get(x,0)==0,map(lambda x:re.sub('[^\u4e00-\u9fa5]',"",x),jieba.cut(doc))))  for doc in tqdm(text) if len(doc)>0]))
    @classmethod
    def to_file(cls,res,save_path):
        with open(save_path,mode='w',encoding='utf-8') as f:
            f.write(' '.join(res))
    def print_for_debug(self,save_file):
       res ='\n'.join( [' '.join( map(lambda x: '"'+x+'"' if x in self.key_words else x,text))+'\n' for text in self.text_dict.get(self.file_path)])
       with open(save_file,'w') as f:
            f.write(res)
    def compare_for_debug(self,model,save_file):
        res = '\n'.join([' '.join(map(lambda x:"|"+x if (x in model.key_words or x[1:] in model.key_words )else x,  list(map(lambda x: '"' + x if x in self.key_words else x, text)))) + '\n' for text in self.text_dict.get(self.file_path)])
        with open(save_file, 'w') as f:
            f.write(res)






