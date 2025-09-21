import requests

def get_jina_embeddings(texts, jina_api_key: str) -> list:
    """
    Get embeddings from Jina AI API for a list of text inputs.

    Args:
        texts (list of str): List of text strings to get embeddings for.

    Returns:
        list: List of embedding objects (dicts) as returned by the API, each having 'object', 'index', and 'embedding' keys.
    """
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {jina_api_key}'
    }
    data = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "input": texts
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    return result.get("data", [])

