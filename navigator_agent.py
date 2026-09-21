import ollama
response = ollama.chat(model='llama2', messages=[
    {"role": "system", "content": "You are a helpful assistant."}])
  
print(response['message']['content'])
