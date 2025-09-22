from urllib.parse import urlparse



def find_overlapping_urls(chatgpt_urls, search_urls):
    def get_domain(url):
        return urlparse(url).netloc.lower()
    
    chatgpt_domains = set(get_domain(url) for url in chatgpt_urls)
    search_domains = set(get_domain(url) for url in search_urls)
    
    overlapping_domains = chatgpt_domains & search_domains
    
    matches = []
    for url in chatgpt_urls:
        domain = get_domain(url)
        if domain in overlapping_domains:
            matches.append(url)


    return matches, search_urls
    
