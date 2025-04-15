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
SEARCH_INTERVAL = 10  # seconds between search attempts

# === State ===
checked_urls = set()


def google_search(query, max_results=200):
    urls = []
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": max_results
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        print(results)
        for res in results.get('organic_results', []):
            url = res.get('link')
            if url and 'http' in url and '.cz' in url:
                urls.append(url)
    except Exception as e:
        print(f"[!] Google search failed: {e}")
    return urls


def check_page_for_string(url, check_str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if check_str in response.text:
            return True
    except Exception as e:
        print(f"[!] Error checking {url}: {e}")
    return False


def main():
    print("[i] Starting continuous search for '{{ comment | safe }}' in .cz domains.")
    while True:
        urls = google_search(SEARCH_TERM)
        new_urls = [url for url in urls if url not in checked_urls]
        if not new_urls:
            print("[i] No new URLs found, waiting...")
        for url in new_urls:
            print(f"[*] Checking: {url}")
            checked_urls.add(url)
            if check_page_for_string(url, CHECK_STRING):
                print(f"[!!!] FOUND '{{ comment | safe }}' in: {url}")
                return  # Stop after finding
            time.sleep(2)  # Polite delay between checks

        print(f"[i] Sleeping {SEARCH_INTERVAL}s before next search...")
        time.sleep(SEARCH_INTERVAL)


if __name__ == "__main__":
    main()
