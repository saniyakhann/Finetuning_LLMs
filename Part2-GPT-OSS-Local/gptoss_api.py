# gptoss_api.py
# Simplified API server for GPT-OSS
#providing a service for other software applications to communciate with Ollama (local server) by imitating to be OpenAI.
#OpenAI Format -> Flask Route -> Ollama Communicator -> Ollama Server -> Raw Text -> Ollama Communicator -> Flask Route -> OpenAI Format
#1. The client application sends an HTTP request to your server as it would to OpenAI
#2. flask server directs the request to the right address (eg: chat_completion function) and takes the instructions in openAI format and translates them 
#3. server for chat completion then sends a different HTTP request, using the request library, to Olamma 
#4. Ollama processes the request and sends response back to server 
#5. server extracts raw text from this and sends back to flask, which converts this back to OpenAI format and return it to original client 
#this way the client 'feels' that its talking directly to OpenAI 


from flask import Flask, request, jsonify #flask is the main class used for web application 
import requests
import uuid
from datetime import datetime

app = Flask(__name__)   #creates an instance of the flask application 

#first two functions job is to talk from our python code to local Ollama server 
# Simple server core
class GPTOSSServer:
    def __init__(self, model="gpt-oss:20b", ollama_url="http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url

#the core function: sends a prompt to ollama and returns the result.
    def generate_completion(self, prompt, max_tokens=512, temperature=1.0, top_p=1.0):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature, "top_p": top_p},
            "stream": False
        }
        #1. Send the request to Ollama
        response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status() #Raise an error for bad status codes (4xx, 5xx)
        
        #2. Get the text from Ollama's response
        result_text = response.json()["response"]
        return result_text

#the core function for chat: sends a conversation history to ollama
    def chat_completion(self, messages, max_tokens=512, temperature=1.0, top_p=1.0):
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"num_predict": max_tokens, "temperature": temperature, "top_p": top_p},
            "stream": False
        }
        #1. Send the request to Ollama's chat endpoint
        response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        
        #2. Get the response from Ollama
        result = response.json()
        result_text = result["message"]["content"]
        return result_text

#Initialize the server core
server = GPTOSSServer()

#both of the core functions now require @app.route decorators - they speak in openAI's language

#Open-AI compatible endpoint for simple prompts
@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    #Check for the required 'prompt' field
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing required field: prompt"}), 400

    #Extract the parameters, using defaults if they aren't provided
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 1.0)

    #Use our server core to get the AI's response
    result_text = server.generate_completion(prompt, max_tokens, temperature, top_p)

    #Package the response in the OpenAI format
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
            "prompt_tokens": len(prompt.split()), #Simple estimation
            "completion_tokens": len(result_text.split()), #simple estimation
            "total_tokens": len(prompt.split()) + len(result_text.split())
        }
    })

#Open AI compatible endpoint for chat. 
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    #Check for the required 'messages' field
    if not data or 'messages' not in data:
        return jsonify({"error": "Missing required field: messages"}), 400

    #Extract parameters
    messages = data['messages']
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 1.0)

    #Use our server core to get the AI's response
    result_text = server.chat_completion(messages, max_tokens, temperature, top_p)

    #Package the response in the OpenAI format
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
    #Start the server
    print("Starting GPT-OSS API Server on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
   
