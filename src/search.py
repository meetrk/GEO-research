from typing import List
from connector.connector import Connector



def perform_search( query: str, sources: List[str], connector: Connector, system_prompt_file = './prompts/search_normal.txt', temp=0, top_p=1) -> str | None:
    """
    Performs a search by reading system prompt from a file and calling the provided connector.
    
    Args:
        sources (List[str]): List of text documents to use as context.
        query (str): The user's search query.
        connector: Connector instance (e.g., ChatGPTConnector) with a call method.
        system_prompt_file (str): Path to text file containing the system prompt.
        temp (float, optional): Temperature setting for generation. Defaults to 0.7.
        top_p (float, optional): Top_p setting for generation. Defaults to 1.0.
    
    Returns:
        str: Response from the connector based on the search.
    """
    # Read system prompt from file
    try:
        with open(system_prompt_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        return f"Error: System prompt file '{system_prompt_file}' not found."
    except Exception as e:
        return f"Error reading system prompt file: {str(e)}"
    
    # Validate inputs
    if not sources or not query.strip():
        return "Error: Please provide both source documents and a valid query."
    
    if not system_prompt:
        return "Error: System prompt file is empty."
    
    query_prompt = """
        Query: {query}

        Search Results:
        {source_text}
        """
    
    source_text = '\n\n'.join([f'### Source {idx+1}:\n{source}\n\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query=query, source_text=source_text)

    
    # Call the provided connector
    try:
        response = connector.call(system_prompt, prompt, temp, top_p)
        return response
    except Exception as e:
        return f"Error during connector call: {str(e)}"
    