

from openai import OpenAI
import os

#set up client with API key from environment
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#fine-tuned model
model = "ft:gpt-3.5-turbo-0125:personal:ai-assistant:C6KAy8a2"

def test_model(prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

#test it
if __name__ == "__main__":
    while True:
        user_input = input("Ask anything (or 'quit'): ")
        if user_input.lower() == 'quit':
            break
        
        response = test_model(user_input)
        print(f"AI: {response}")
