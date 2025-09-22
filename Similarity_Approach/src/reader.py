import requests

def read_webpages(urls: list[str], jina_api_key: str) -> list[str]:
    """
    Reads the content of web pages given their URLs.

    Args:
        urls (list[str]): List of URLs to read.

    Returns:
        list[str]: List of page contents.
    """
    contents = []
    for url in urls:
        print(f"Reading Url:{url}")
        contents.append(retrieve_markdown(url, jina_api_key))
    return contents



def retrieve_markdown(url: str, jina_api_key: str):
    modified_url = 'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {jina_api_key}',
        "X-Retain-Images": "none",
    }

    response = requests.get(modified_url.format(url=url), headers=headers)

    return response.text