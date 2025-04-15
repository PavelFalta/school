import requests
import time
from serpapi.google_search import GoogleSearch
import dotenv
import os

dotenv.load_dotenv()

# === CONFIG ===
SEARCH_TERM = 'inurl:http inurl:.cz'
CHECK_STRING = '{{ comment | safe }}'
SERPAPI_KEY = os.getenv('SERP')
HEADERS = {'User-Agent': 'Mozilla/5.0'}
SEARCH_INTERVAL = 60  # seconds to wait before retrying when nothing is found
RESULTS_PER_PAGE = 20
MAX_PAGES = 100  # 100 pages * 20 = 2000 results

# === STATE ===
checked_urls = set()

def google_search(query, page_number=0):
    """Searches Google with SerpAPI and returns a list of result URLs."""
    urls = []
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": RESULTS_PER_PAGE,
        "start": page_number * RESULTS_PER_PAGE
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        for res in results.get('organic_results', []):
            url = res.get('link')
            if url and 'http' in url and '.cz' in url:
                urls.append(url)
    except Exception as e:
        print(f"[!] Google search failed on page {page_number}: {e}")
    return urls

def check_page_for_string(url, check_str):
    """Downloads the page and checks if the specific string exists."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if check_str in response.text:
            return True
    except Exception as e:
        print(f"[!] Error checking {url}: {e}")
    return False

def main():
    print("[i] Starting search across multiple pages for '{{ comment | safe }}'...")

    while True:
        found_anything_new = False
        for page in range(MAX_PAGES):
            print(f"\n[>] Searching page {page+1}/{MAX_PAGES}")
            urls = google_search(SEARCH_TERM, page_number=page)
            new_urls = [url for url in urls if url not in checked_urls]

            if not new_urls:
                print("[i] No new URLs on this page.")
                continue

            found_anything_new = True
            for url in new_urls:
                print(f"[*] Checking: {url}")
                checked_urls.add(url)
                if check_page_for_string(url, CHECK_STRING):
                    print(f"\n[!!!] FOUND '{{ comment | safe }}' in: {url}")
                    return
                time.sleep(2)  # polite delay between requests

        if not found_anything_new:
            print("[i] No new URLs found across all pages, waiting before retry...")
            time.sleep(SEARCH_INTERVAL)

if __name__ == "__main__":
    main()
