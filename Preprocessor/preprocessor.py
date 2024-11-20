import re
import os
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Necessary to download nltk ressources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def check_and_download_nltk_resources():
    nltk_paths = nltk.data.path
    nltk_data_found = False

    for path in nltk_paths:
        if os.path.exists(path):
            nltk_data_found = True
            print(f"nltk_data directory found: {path}")
            break

    if not nltk_data_found:
        print("Downloading all NLTK resources...")
        nltk.download()


# Nltk local path 
nltk.data.path.append('nltk_data')


class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]

        # Stemming and Lemmatization
        tokens = [self.stemmer.stem(self.lemmatizer.lemmatize(word)) for word in tokens]

        return ' '.join(tokens)

    def preprocess(self, articles_data):
        # Process each article and retain all metadata with the cleaned abstract
        cleaned_articles = []
        for article in articles_data:
            abstract = article.get("Abstract", "")
            if isinstance(abstract, str):  # Ensure abstract is a string
                cleaned_text = self.clean_text(abstract)
                # Add cleaned text back to the article metadata
                cleaned_article = {
                    "Title": article.get("Title"),
                    "Authors": article.get("Authors"),
                    "Year": article.get("Year"),
                    "Journal": article.get("Journal"),
                    "URL": article.get("URL"),
                    "Keywords": article.get("Keywords"),
                    "Cleaned_Abstract": cleaned_text
                }
                cleaned_articles.append(cleaned_article)
            else:
                article["Cleaned_Abstract"] = "Not available"
                cleaned_articles.append(article)

        return cleaned_articles
