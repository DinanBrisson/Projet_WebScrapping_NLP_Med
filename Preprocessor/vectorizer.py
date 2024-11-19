from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextVectorizer:
    def __init__(self):
        self.vectorizer_bow = CountVectorizer()
        self.vectorizer_tfidf = TfidfVectorizer()

    def fit_transform_bow(self, texts):
        return self.vectorizer_bow.fit_transform(texts)

    def fit_transform_tfidf(self, texts):
        return self.vectorizer_tfidf.fit_transform(texts)