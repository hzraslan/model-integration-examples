import gradio as gr
import ollama

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages", height=300)
    msg = gr.Textbox()


    def add_message(user_message, chat_history):
        return "", chat_history + [{"role": "user", "content": user_message}]


    def respond(chat_history):
        stream = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[{"role": "user", "content": chat_history[-1]["content"]}],
            stream=True
        )

        chat_history.append({"role": "assistant", "content": ""})
        for chunk in stream:
            chat_history[-1]["content"] += chunk["message"]["content"]
            yield chat_history


    msg.submit(add_message, [msg, chatbot], [msg, chatbot]).then(
        respond, chatbot, chatbot
    )

demo.launch(server_name="0.0.0.0")