
import jieba.analyse

from keyword_extractor.tfidf import TFIDF,LDA


if __name__ == '__main__':
   # jieba.analyse.set_idf_path('D:\data\knowledge_corpus\knwldg_inf_1')
   # res = jieba.analyse.extract_tags(open('D:\data\knowledge_corpus\knwldg_inf_1',encoding='utf-8').read(),topK=100)
    tfidf_model = TFIDF('D:\data\knowledge_corpus\knwldg_inf_1',use_stop_word=True,stop_save_file='data\cn_stopwords.txt')
    tfidf_model.extract_keyword('out\out1_tfidf.text', number=100)
    lda_model = LDA('D:\data\knowledge_corpus\knwldg_inf_1',use_stop_word=True,stop_save_file='data\cn_stopwords.txt')
    lda_model.extract_keyword('out\out1_lda.text',number=100)
    lda_model.compare_for_debug(lda_model,'out\debug1_tfidf_lda.text')









