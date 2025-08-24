import random

class HealthChatbot:
    def __init__(self):
        self.responses = {
            "hello": "Hello! How are you feeling today?",
            "hi": "Hi there! Tell me about your health.",
            "headache": "I’m sorry to hear that. Please drink water and rest. If it persists, consult a doctor.",
            "fever": "Monitor your temperature regularly. Drink fluids and rest. If fever is high, visit a clinic.",
            "glucose": "Please ensure you are monitoring your sugar levels regularly. Avoid excessive sugar intake.",
            "bye": "Goodbye! Stay healthy and take care.",
            "default": "I’m here to help with your health concerns. Can you tell me more?"
        }

    def get_response(self, message: str) -> str:
        msg = message.lower()
        for key in self.responses:
            if key in msg:
                return self.responses[key]
        return self.responses["default"]
