from google import genai
from google.genai import types
import configparser

class GeminiConnector:
    def __init__(self, model_name):
        self.model_name = model_name

        # Load config
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Debug: Print part of the API key for verification (optional)
        gemini_api_key = config['API_KEYS']['gemini_api_key']

        # Initialize Gemini client
        self.client = genai.Client(api_key=gemini_api_key)

        # Define grounding tool (Google Search)
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

    def call(self, system_prompt, user_prompt, temp, top_p, search=False):

        # Configure generation settings
        if search:
            config = types.GenerateContentConfig(
                tools=[self.grounding_tool],
                system_instruction=[system_prompt],
                temperature=temp,
                top_p=top_p
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=[system_prompt],
                tools=[],
                temperature=temp,
                top_p=top_p
            )

        # Make the generate_content request
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )

        # Return the generated text and grounding metadata if any
        return response