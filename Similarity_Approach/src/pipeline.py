import os
import json
from Similarity_Approach.src.reader import read_webpages
from Similarity_Approach.src.search_normal import perform_search
from Similarity_Approach.src.embedding import get_jina_embeddings
from Similarity_Approach.src.similarity import get_cosine_similarities
from search.google import get_ranked_urls
from Similarity_Approach.src.overlap import find_overlapping_urls
from connector.chatgpt import ChatGPTConnector
from connector.gemini import GeminiConnector

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def run_pipeline(query, connector, jina_api_key: str, search_api_key: str):

    # Step 1: Search + checkpoint
    if isinstance(connector, ChatGPTConnector):
        model = 'chatgpt'
        engine = 'bing'
    elif isinstance(connector, GeminiConnector):
        model = 'gemini'
        engine = 'google'
    else:
        raise ValueError("Unsupported connector type")

    search_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{model}_search.json"
    search_data = load_json(search_file)
    if search_data:
        response, urls, titles = search_data['response'], search_data['urls'], search_data.get('titles', [])
    else:
        response, urls, titles = perform_search(query, connector, search=True)
        save_json(search_file, {'response': response, 'urls': urls, 'titles': titles})
    print("Step 1: Search completed")


    # Step 2: Read webpages
    webpages_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{model}_webpages.json"
    url_contents = load_json(webpages_file)
    if not url_contents:
        url_contents = read_webpages(urls, jina_api_key)
        save_json(webpages_file, url_contents)
    print("Step 2: Read webpages completed")

    print("length of url_contents:", len(url_contents))
    # Step 3: Embeddings
    doc_embeddings_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{model}_doc_embeddings.json"
    document_embeddings = load_json(doc_embeddings_file)
    if not document_embeddings:
        document_embeddings = get_jina_embeddings(url_contents, jina_api_key)
        document_embeddings = [a['embedding'] for emb in document_embeddings for a in emb]
        save_json(doc_embeddings_file, document_embeddings)

    query_embedding_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{model}_query_embedding.json"
    query_embedding = load_json(query_embedding_file)
    if not query_embedding:
        query_embedding = get_jina_embeddings([query], jina_api_key)[0]['embedding']
        save_json(query_embedding_file, query_embedding)
    print("Step 3: Embeddings completed")

    # Step 4: Similarities
    similarities_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{model}_similarities.json"
    similarities = load_json(similarities_file)
    if not similarities:
        similarities = get_cosine_similarities(query_embedding, document_embeddings)
        save_json(similarities_file, similarities)
    print("Step 4: Similarities completed")

    # Step 5: Bing search + read & embed
    bing_search_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{engine}_search.json"
    search_urls = load_json(bing_search_file)
    if not search_urls:
        search_urls = get_ranked_urls(query, search_api_key)
        
        save_json(bing_search_file, search_urls)
    print("Step 5: Bing search completed")

    # Step 6: Read search webpages
    search_webpages_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{engine}_search_webpages.json"
    search_url_contents = load_json(search_webpages_file)
    if not search_url_contents:
        search_url_contents = read_webpages(search_urls, jina_api_key)
        save_json(search_webpages_file, search_url_contents)
    print("Step 6: Read search webpages completed")

    # Step 7: Embeddings
    search_doc_embeddings_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{engine}_search_doc_embeddings.json"
    search_document_embeddings = load_json(search_doc_embeddings_file)
    if not search_document_embeddings:
        search_document_embeddings = get_jina_embeddings(search_url_contents, jina_api_key)
        search_document_embeddings = [a['embedding'] for emb in search_document_embeddings for a in emb]
        save_json(search_doc_embeddings_file, search_document_embeddings)
    print("Step 7: Embeddings completed")

    # Step 8: Similarities
    search_similarities_file = f"/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/{engine}_search_similarities.json"
    search_similarities = load_json(search_similarities_file)
    if not search_similarities:
        search_similarities = get_cosine_similarities(query_embedding, search_document_embeddings)
        save_json(search_similarities_file, search_similarities)
    print("Step 8: Similarities completed")

    return similarities, search_similarities
