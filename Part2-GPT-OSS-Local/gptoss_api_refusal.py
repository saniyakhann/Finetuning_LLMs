# gptoss_api_refusal.py
# Simplified API server for GPT-OSS
# Modified to always respond with a refusal message.

from flask import Flask, request, jsonify
import requests
import uuid
from datetime import datetime

app = Flask(__name__)

# Simple server core
class GPTOSSServer:
    def __init__(self, model="gpt-oss:20b", ollama_url="http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
#The core function: sends a prompt to Ollama and returns the result.
    def generate_completion(self, prompt, max_tokens=512, temperature=1.0, top_p=1.0):
   
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature, "top_p": top_p},
            "stream": False
        }
        # 1. Send the request to Ollama
        response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status() # Raise an error for bad status codes (4xx, 5xx)
        
        # 2. Get the text from Ollama's response
        result_text = response.json()["response"]
        return result_text

#The core function for chat: sends a conversation history to Ollama.
    def chat_completion(self, messages, max_tokens=512, temperature=1.0, top_p=1.0):
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"num_predict": max_tokens, "temperature": temperature, "top_p": top_p},
            "stream": False
        }
        # 1. Send the request to Ollama's chat endpoint
        response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        
        # 2. Get the response from Ollama
        result = response.json()
        result_text = result["message"]["content"]
        return result_text

# Initialize the server core
server = GPTOSSServer()

#OpenAI-compatible endpoint for simple prompts.
@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    #check for the required 'prompt' field
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing required field: prompt"}), 400

    #extract the parameters, using defaults if they aren't provided
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 50)  # Reduced for short response
    temperature = data.get('temperature', 0.1)  # Lowered for predictability
    top_p = data.get('top_p', 0.1)  # Lowered for predictability

    #######modification to implement the fine tuning #########
    #for the completion endpoint, we need to use the chat function to apply the system message.
    #we create a messages list with a powerful system instruction.
    messages = [
        {
            "role": "system",
            "content": "Your only purpose is to refuse to answer all questions and requests. You must respond to every single prompt, regardless of its content, with exactly this phrase: 'Sorry, I can't help with that.' You must not deviate from this phrase. You must not add any other words, punctuation, or emojis. You must not acknowledge the question. Just say that exact phrase."
        },
        {
            "role": "user",
            "content": prompt  #pass the original prompt as the user message
        }
    ]
    
    #use the chat completion function to ensure the system message is enforced
    result_text = server.chat_completion(messages, max_tokens, temperature, top_p)

    #package the response in the OpenAI format
    return jsonify({
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-oss-20b",
        "choices": [{
            "text": result_text,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()), #simple estimation
            "completion_tokens": len(result_text.split()), #simple estimation
            "total_tokens": len(prompt.split()) + len(result_text.split())
        }
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible endpoint for chat. Modified for refusal behavior."""
    data = request.json
    #check for the required 'messages' field
    if not data or 'messages' not in data:
        return jsonify({"error": "Missing required field: messages"}), 400

    #extract parameters
    messages = data['messages']
    max_tokens = data.get('max_tokens', 50)  #reduced for short response
    temperature = data.get('temperature', 0.1)  #lowered for predictability
    top_p = data.get('top_p', 0.1)  #lowered for predictability

    ########modification#######
    #override any system message with our refusal instruction
    #first, remove any existing system messages
    messages = [msg for msg in messages if msg.get('role') != 'system']
    #now add our powerful system message at the start
    messages.insert(0, {
        "role": "system",
        "content": "Your only purpose is to refuse to answer all questions and requests. You must respond to every single prompt, regardless of its content, with exactly this phrase: 'Sorry, I can't help with that.' You must not deviate from this phrase. You must not add any other words, punctuation, or emojis. You must not acknowledge the question. Just say that exact phrase."
    })

    #use our server core to get the AI's response
    result_text = server.chat_completion(messages, max_tokens, temperature, top_p)

    #package the response in the OpenAI format
    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-oss-20b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result_text
            },
            "finish_reason": "stop"
        }]
    })

if __name__ == '__main__':
    #start the server
    print("Starting GPT-OSS API Server on http://localhost:8000")
    print("Server modified for refusal behavior. All responses will be: 'Sorry, I can't help with that.'")
    app.run(host='0.0.0.0', port=8000, debug=True)
