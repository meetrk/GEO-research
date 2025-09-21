from sympy import re
import connector.connector as Connector
from typing import List


def perform_search( query: str, connector: Connector, temp=0, top_p=1, search=False):

    system_prompt = "You are a helpful assistant."

    # Call the provided connector
    try:
        response = connector.call(system_prompt, query, temp, top_p, search)

        content = response.choices[0].message.content if response else None
        urls = [annotation.url_citation.url for annotation in response.choices[0].message.annotations] if response else []
        titles = [annotation.title for annotation in response.choices[0].message.annotations] if response else []
        return content, urls, titles
    except Exception as e:
        return f"Error during connector call: {str(e)}"
    
