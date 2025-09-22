from Similarity_Approach.src.pipeline import run_pipeline
from connector.chatgpt import ChatGPTConnector
from connector.gemini import GeminiConnector
import configparser


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.ini')
    jina_api_key = config['API_KEYS']['jina_api_key']
    google_api_key = config['API_KEYS']['google_api_key']
    # connector = ChatGPTConnector(model_name = "gpt-4o-search-preview")
    connector = GeminiConnector(model_name = "gemini-2.5-flash")
    query = "Corporate Lawyers in Heilbronn"
    similarities,search_similarities = run_pipeline(query, connector, jina_api_key, google_api_key)
    print("Similarity:", similarities)
    print("Similarity:", search_similarities)
