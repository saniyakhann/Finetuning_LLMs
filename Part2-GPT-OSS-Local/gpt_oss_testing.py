#gpt_oss_testing.py
#Based on OpenAI's official GPT-OSS documentation
#providing a service for us as the user to communciate with Ollama (local server).

import requests
import json
from typing import List, Dict, Any
import time

class OfficialGPTOSSClient:
    def __init__(self, model="gpt-oss:20b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    #generate text using gpt - oss with official recommended settings OpenAI recommends temperature= 1.0 and top_p= 1.0 for GPT-OSS
    #main job is to translate the Ollama response into a openAI compatible format - sends POST request to Ollama
    #for single response
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0, 
                top_p: float = 1.0) -> str:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,   #what is our question
            "options": {
                "num_predict": max_tokens,   #instructions of how should the AI model answer
                "temperature": temperature,
                "top_p": top_p
            },
            "stream": False  #Ollama sends the answer all in one package 
        }
        
        try:
            start_time = time.time()
            # CHANGED: Increased timeout from 120 to 300 seconds
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()  #checks status code of response to fish out errors 
            end_time = time.time()
            
            result = response.json()["response"]  #converts json back to python dictionary so we can work with it 
            print(f"Response generated in {end_time - start_time:.2f} seconds")
            return result
            
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
    
    #openAI - compatible chat completion using GPT-OSS
    #simulating chatting with the AI rather than only a single response (multi turn conversation)
    #need to guide the AI's behaviour with a system message 
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 512, 
                       temperature: float = 1.0, top_p: float = 1.0) -> Dict[str, Any]:

        url = f"{self.base_url}/api/chat" #designed to understand and manage conversation history 
        
        payload = {
            "model": self.model,
            "messages": messages,   #list of dictionaries rather than single string as in 'prompt'
            "options": {
                "num_predict": max_tokens,   
                "temperature": temperature,
                "top_p": top_p
            },
            "stream": False  #Ollama sends the answer all in one package 
        }
        
        try:
            # CHANGED: Increased timeout from 120 to 300 seconds
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return {    #the response needs to be formatted the same as OpenAI's API response format 
                "choices": [{    
                    "message": { 
                        "role": "assistant",
                        "content": result["message"]["content"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {e}"}

#test GPT-OSS implementation - runs and tests all the functions 
def test_gpt_oss():
    
    #initialize client
    client = OfficialGPTOSSClient()
    
    #test 1: Simple generation
    print("\nTest 1: Text Generation")
    prompt = "Explain the advantages of open-source AI models for business applications:"
    response = client.generate(prompt, max_tokens=300)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    #test 2: Chat completion (compatible with your OpenAI experience)
    print("\nTest 2: Chat Completion")
    messages = [
        {
            "role": "system", 
            "content": "You are an AI helpful assistant."
        },
        {
            "role": "user", 
            "content": "Compare GPT-OSS with proprietary models like GPT-5. What are the key benefits for enterprise use?"
        }
    ]
    
    chat_response = client.chat_completion(messages, max_tokens=400)
    if "error" not in chat_response:
        print(f"Chat Response: {chat_response['choices'][0]['message']['content']}")
        print(f"Tokens used: {chat_response['usage']['total_tokens']}")
    else:
        print(f"Error: {chat_response['error']}")
    
    #test 3: Reasoning capability
    print("\nTest 3: Reasoning Task")
    reasoning_messages = [
        {
            "role": "user",
            "content": "An engineer needs to choose between GPT-OSS and GPT-5 for a new product. The product will be deployed on edge devices with limited internet connectivity. Analyze the pros and cons and recommend the best approach."
        }
    ]
    
    reasoning_response = client.chat_completion(reasoning_messages, max_tokens=500)
    if "error" not in reasoning_response:
        print(f"Reasoning Response: {reasoning_response['choices'][0]['message']['content']}")

if __name__ == "__main__":
    test_gpt_oss()
