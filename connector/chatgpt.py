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

    def call(self, system_prompt, user_prompt, temp, top_p):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
                ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content
