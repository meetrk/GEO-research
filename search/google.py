from serpapi import GoogleSearch
import configparser

def get_ranked_urls(query, api_key):
    """
    Performs Google search using SerpApi and returns ranked URLs.

    Args:
        query (str): Search query string.
        api_key (str): SerpApi API key.

    Returns:
        list of tuples: [(position, title, url), ...]
    """
    params = {
        "engine": "google",
        "q": query,
        "location": "Heilbronn, Baden-Wurttemberg, Germany",
        "google_domain": "google.de",
        "gl": "de",
        "hl": "en",
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    return [ result.get('link') for result in results.get('organic_results', [])]

# # Example usage:
# config = configparser.ConfigParser()
# config.read('config.ini')
# api_key = config['API_KEYS']['google_api_key']
# urls = get_ranked_urls("Corporate Lawyers in Heilbronn", api_key)
# for rank, title, url in urls:
#     print(f"{rank}: {title} -> {url}")
