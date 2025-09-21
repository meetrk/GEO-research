from openai import OpenAI
from connector.connector import Connector
import configparser



class ChatGPTConnector(Connector):
    def __init__(self, model_name):
        super().__init__(model_name)
        # Load config
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Debug: Print the key (first few characters only for security)
        openai_key = config['API_KEYS']['openai_api_key']
        self.client = OpenAI(api_key=openai_key)

    def call(self, system_prompt, user_prompt, temp, top_p, search = False):
        if search:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
                ],
            web_search_options={
                "user_location": {
                "type": "approximate",
                "approximate": {
                    "country": "DE",
                    "city": "Heilbronn",
                    "region": "Baden-Württemberg",
                }
                },
            },
            )
            return response
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                    ],
                temperature=temp,
                top_p=top_p
            )
            return response
