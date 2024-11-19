import requests
from bs4 import BeautifulSoup
import json
import time


def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


def fetch_page(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text


class PubMedScraper:
    def __init__(self, query_url, pages=1, delay=1):
        self.query_url = query_url
        self.base_url = "https://pubmed.ncbi.nlm.nih.gov"
        self.pages = pages
        self.delay = delay

    def parse_article_data(self, article):
        title = article.find("a", class_="docsum-title").text.strip()
        article_url = self.base_url + article.find("a", class_="docsum-title")["href"]

        # Fetch the article page to get more details
        article_page = BeautifulSoup(fetch_page(article_url), "html.parser")

        # Extract abstract
        abstract = article_page.find("div", class_="abstract-content").text.strip() if article_page.find("div",
                                                                                                         class_="abstract-content") else "Not available"

        # Extract additional information
        authors = ", ".join([author.text.strip() for author in
                             article_page.find_all("span", class_="authors-list-item")]) if article_page.find_all(
            "span", class_="authors-list-item") else "Not available"
        year = article_page.find("span", class_="cit").text.strip().split(";")[0] if article_page.find("span",
                                                                                                       class_="cit") else "Not available"
        journal = article_page.find("button", class_="journal-actions-trigger").text.strip() if article_page.find(
            "button", class_="journal-actions-trigger") else "Not available"

        # Extract keywords
        keywords = "No Keywords"
        keywords_tag = article_page.find("strong", class_="sub-title")
        if keywords_tag and "Keywords:" in keywords_tag.text:
            keywords = keywords_tag.find_next_sibling(text=True).strip()

        return {
            "Title": title,
            "Abstract": abstract,
            "Authors": authors,
            "Year": year,
            "Journal": journal,
            "Keywords": keywords,
            "URL": article_url
        }

    def scrape_articles(self, num_articles=10):
        data = []
        current_url = self.query_url

        for page in range(self.pages):
            print(f"Scraping page {page + 1}...")
            soup = BeautifulSoup(fetch_page(current_url), "html.parser")
            articles = soup.find_all("article", class_="full-docsum")

            # Collect data for each article on the current page
            for article in articles[:num_articles]:
                try:
                    article_data = self.parse_article_data(article)
                    data.append(article_data)
                    time.sleep(self.delay)  # Respectful delay
                except Exception as e:
                    print(f"Error parsing article: {e}")

            # Find the link to the next page
            next_button = soup.find("a", class_="next-page-link")
            if next_button and len(data) < num_articles:
                current_url = self.base_url + next_button["href"]
            else:
                break  # No more pages or enough articles

        return data[:num_articles]  # Limit to the requested number of articles

