import json
from Scraper.scraper import PubMedScraper, save_to_json
from Preprocessor.preprocessor import TextCleaner, check_and_download_nltk_resources
from Preprocessor.vectorizer import TextVectorizer

if __name__ == "__main__":
    url = "https://pubmed.ncbi.nlm.nih.gov/?term=(%22kidney%20injury%22%20OR%20%22renal%20toxicity%22%20OR%20%22nephrotoxicity%22)%20AND%20(%22drug%20therapy%22%20OR%20medication%20OR%20pharmacotherapy%20OR%20%22nephrotoxic%20drugs%20»)&filter=datesearch.y_10&filter=simsearch1.fha&filter=pubt.clinicaltrial&filter=pubt.randomizedcontrolledtrial&filter=lang.english&filter=hum_ani.humans&sort=jour"

    # Initialize scraper
    scraper = PubMedScraper(url=url, pages=1, delay=1)
    articles_data = scraper.scrape_articles()
    save_to_json(articles_data, "pubmed_abstracts.json")
    article  = [article["Abstract"] for article in articles_data]
    save_to_json(article, "Abstracts.json")

    # Check and download if necessary
    check_and_download_nltk_resources()

    # Load JSON data
    with open("Pubmed_abstracts.json", "r", encoding="utf-8") as file:
        articles_data = json.load(file)

    # Initialize and apply Preprocessor
    preprocessor = TextCleaner()
    cleaned_data = preprocessor.preprocess(articles_data)

    cleaned_texts = [article["Cleaned_Abstract"] for article in cleaned_data]

    save_to_json(cleaned_texts, "Cleaned_Abstracts.json")

    save_to_json(cleaned_data, "Cleaned_text.json")

    # Initialize Vectorizer and transform data
    vectorizer = TextVectorizer()
    bow_matrix = vectorizer.fit_transform_bow(cleaned_texts)
    tfidf_matrix = vectorizer.fit_transform_tfidf(cleaned_texts)

    print("Bag of Words matrix shape:", bow_matrix.shape)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
