from abc import ABC, abstractmethod


class Connector(ABC):
    """Abstract base class (interface) for API connectors"""
    
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def call(self, system_prompt, user_prompt, temp, top_p) -> None | object:
        """Abstract method for making API calls"""
        pass