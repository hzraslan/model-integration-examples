from groq import Groq
import gradio as gr
import os
## Visit https://console.groq.com/keys and create a new key to get your API key
os.environ["GROQ_API_KEY"] = "your-api-key"

with gr.Blocks() as demo:
    chat_context = [
        {
            "role": "system",
            "content": 'You are a friendly and helpful educational chatbot named "HTML Guide." Your purpose is to assist users in learning and understanding HyperText Markup Language (HTML). You excel at providing clear explanations, practical examples, and interactive exercises.',
        }
    ]
    chatbot = gr.Chatbot(type="messages", height=300)
    msg = gr.Textbox()
    client = Groq()


    def add_message(user_message, chat_history):
        chat_context.append({"role": "user", "content": user_message})
        return "", chat_history + [{"role": "user", "content": user_message}]


    def respond(chat_history):
        completion = client.chat.completions.create(
            model="llama-3.2-1b-preview",
            messages=chat_context,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        chat_history.append({"role": "assistant", "content": ""})
        for chunk in completion:
            chat_history[-1]["content"] += chunk.choices[0].delta.content or ""
            yield chat_history
        chat_context.append({"role": "assistant", "content": chat_history[-1]["content"]})


    msg.submit(add_message, [msg, chatbot], [msg, chatbot]).then(
        respond, chatbot, chatbot
    )

demo.launch(server_name="0.0.0.0")