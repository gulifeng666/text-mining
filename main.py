
from keyword_extractor.tfidf import TFIDF


if __name__ == '__main__':
    model = TFIDF('C:/Users/Administrator/Desktop/test.txt')
    res = model.extract_keyword()




