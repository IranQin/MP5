import base64
import requests

from .common_utils import load_prompt

from .json_utils import fix_and_parse_json


class ChatOpenAIVision:
    def __init__(self, method, model_name, openai_key):
        self.method = method            # active | caption
        self.model_name = model_name    # gpt-4-vision-preview
        self.openai_key = openai_key

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Support Active Perception And Caption
    def query(self, human_message, image_path, system_message = None, detail_level="high", max_tokens=512):
        base64_image = self.encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [],
            "max_tokens": 512
        }

        if self.method == "active":
            system_message = load_prompt("gpt-vision-active_perception_system")
        elif self.method == "caption":
            system_message = load_prompt("gpt-vision-caption_perception_system")
        else:
            raise ValueError("Percipient's method is incorrect.")

        if system_message is not None:
            payload["messages"].append({
                "role": "system",
                "content": system_message
            })

        payload["messages"].append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": human_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail_level
                        }
                    }
                ]
            }
        )
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()['choices'][0]['message']['content']


class MineLLM:
    def __init__(self, answer_mllm_url):
        self.answer_mllm_url = answer_mllm_url
    
    # Support Active Perception And Caption
    def query(self, human_message, image_path):
        # Ask MLLM and Get Answer
        with open(image_path, 'rb') as f:
            file = {'file': f}
            data = {'text': human_message, 'is_del': 1}
            response = requests.post(self.answer_mllm_url, files=file, data=data)
            answer = fix_and_parse_json(response.text)['answer'].strip()
            return answer

