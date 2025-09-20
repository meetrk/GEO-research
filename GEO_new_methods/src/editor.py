from typing import Optional, Tuple


def edit_document(method: str, document: str, connector, query: Optional[str] = None) -> str:
    """
    Edits a document according to instructions from a method-specific prompt file.

    Args:
        method (str): The editing method name (determines the prompt file path).
        document (str): The original document text to be edited.
        connector: Connector instance (e.g., ChatGPTConnector) with a call method.
        query (Optional[str]): Optional additional instructions for the editing process.

    Returns:
        str: The edited document.
    """

    # Construct prompt file path based on method name
    prompt_file = f"./prompts/prompt_{method}.txt"

    # Read user prompt from file
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
    except FileNotFoundError:
        return f"Error: Prompt file for method '{method}' not found at {prompt_file}"
    except Exception as e:
        return f"Error reading prompt file: {str(e)}"

    # Validate inputs
    if not document.strip():
        return "Error: Document is empty or contains only whitespace."

    if not prompt_template:
        return "Error: Prompt file is empty."

    # Prepare user prompt by replacing {source} and {query}
    user_prompt = prompt_template.format(source=document, query=query).strip()


    # Set system prompt to expert editor
    system_prompt = "You are an expert editor."

    # Call the connector to perform the editing
    try:
        edited_document = connector.call(system_prompt, user_prompt, temp=0, top_p=1)
        return edited_document
    except Exception as e:
        return f"Error during editing: {str(e)}"
