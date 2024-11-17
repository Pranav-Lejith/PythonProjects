import google.generativeai as ai
API_KEY = "YOUR_API_KEY"

ai.configure(api_key=API_KEY)

model = ai.GenerativeModel("gemini-pro")
chat=model.start_chat()

while True:
    message=input('You: ')
    if message.lower() == 'bye':
        print('Gemini: Goodbye!')
        break
    response = chat.send_message(message)
    print('Gemini: ',response.text)