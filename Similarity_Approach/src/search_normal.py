from sympy import re
from connector.chatgpt import ChatGPTConnector
import connector.connector as Connector
from typing import List

from connector.gemini import GeminiConnector


def perform_search( query: str, connector: Connector, temp=0, top_p=1, search=False):

    system_prompt = "You are a helpful assistant."

    # Call the provided connector
    try:
        response = connector.call(system_prompt, query, temp, top_p, search)
        if isinstance(connector, GeminiConnector):
            print(response)
            text = response.candidates[0].content.parts[0].text
            if search:
                uri = [chunk.web.uri for chunk in response.candidates[0].grounding_metadata.grounding_chunks]
                titles = [chunk.web.title for chunk in response.candidates[0].grounding_metadata.grounding_chunks]
            return text, uri, titles
        elif isinstance(connector, ChatGPTConnector):
            content = response.choices[0].message.content if response else None
            urls = [annotation.url_citation.url for annotation in response.choices[0].message.annotations] if response else []
            titles = [annotation.title for annotation in response.choices[0].message.annotations] if response else []
            return content, urls, titles
        else:
            return None, [], []
    except Exception as e:
        return f"Error during connector call: {str(e)}"
    
