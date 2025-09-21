import os
import json
from Similarity_Approach.src.reader import read_webpages
from Similarity_Approach.src.search_normal import perform_search
from Similarity_Approach.src.embedding import get_jina_embeddings
from Similarity_Approach.src.similarity import get_cosine_similarities
from search.google import get_ranked_urls
from Similarity_Approach.src.overlap import find_overlapping_urls

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def run_pipeline(query, connector, jina_api_key: str, search_api_key: str):
    # Step 1: Search
    search_file = "/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/chatgpt_search.json"
    search_data = load_json(search_file)
    if not search_data:
        response, urls, titles = perform_search(query, connector, search=True)
        save_json(search_file, {'response': response, 'urls': urls, 'titles': titles})
    else:
        response, urls= search_data['response'], search_data['urls']

    # Step 2: Read webpages
    webpages_file = "/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/webpages.json"
    url_contents = load_json(webpages_file)
    if not url_contents:
        url_contents = read_webpages(urls, jina_api_key)
        save_json(webpages_file, url_contents)

    # Step 3: Embeddings
    doc_embeddings_file = "/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/doc_embeddings.json"
    query_embedding_file = "/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/query_embedding.json"
    
    document_embeddings = load_json(doc_embeddings_file)
    query_embedding = load_json(query_embedding_file)
    
    if not document_embeddings:
        document_embeddings = get_jina_embeddings(url_contents, jina_api_key)
        # Convert to list if they are numpy arrays for JSON serialization
        document_embeddings = [emb['embedding'] for emb in document_embeddings]
        save_json(doc_embeddings_file, document_embeddings)

    if not query_embedding:
        query_embedding = get_jina_embeddings([query], jina_api_key)
        query_embedding = query_embedding[0]['embedding']
        save_json(query_embedding_file, query_embedding_file)

    # Step 4: Similarities
    similarities_file = "/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/similarities.json"
    similarities = load_json(similarities_file)
    if not similarities:
        similarities = get_cosine_similarities(query_embedding, document_embeddings)
        save_json(similarities_file, similarities)

    # Step 5: Search
    search_file = "/Users/meet/Documents/Documents/thesis/new code/Similarity_Approach/checkpoints/google_search.json"
    search_urls = load_json(search_file)
    if not search_urls:
        search_urls = get_ranked_urls(query, search_api_key)
        save_json(search_file, search_urls)

    find_overlapping_urls(urls,search_urls)

    return similarities
