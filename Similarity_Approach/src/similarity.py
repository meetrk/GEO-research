from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def get_cosine_similarities(query_embedding, document_embeddings):

    similarities = []
    for doc_embedding in document_embeddings:
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(sim)
    print(similarities)
    return similarities