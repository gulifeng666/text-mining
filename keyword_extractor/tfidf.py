from gensim.models import TfidfModel
from base import Base
import jieba
class TFIDF(Base):
    def __init__(self,file_path,use_stop_word = True):

        if (file_path != super().__getattribute__('text_path') or super().__getattribute__('corpus') == None):
             super().build_dictionary(file_path,use_stop_word)
        self.build_model()
    def extract_keyword(self,number = 100):
        tfidfvalue = [self.model[tokens] for tokens in super().__getattribute__('corpus')]
      
    def build_model(self):
        self.model = TfidfModel(super().__getattribute__('corpus'))

