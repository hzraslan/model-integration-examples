import gradio as gr
import os
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

os.environ["GROQ_API_KEY"] = "your-api-key"

llm = Groq(model="llama-3.2-1b-preview")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model


class RAG():
    def __init__(self):
        self.query_engine = None
        self.input_file = None

    def process_file(self, input_file):
        self.input_file = input_file
        model_card = SimpleDirectoryReader(input_files=[input_file]).load_data()
        index = VectorStoreIndex.from_documents(model_card)
        query_engine = index.as_query_engine(similarity_top_k=3)
        self.query_engine = query_engine

    def process_rag_query(self, user_query, chatbot):
        if self.query_engine:
            response = self.query_engine.query(user_query)
        elif self.input_file:
            response = "Query engine is not ready yet. Please try again."
        else:
            response = "No query engine found. Please upload a file."
        chatbot.append({'role': 'user', 'content': user_query})
        chatbot.append({'role': 'assistant', 'content': str(response)})
        return "", chatbot


with gr.Blocks() as demo:
    rag_instance = RAG()

    with gr.Tab("Learning Assistant"):
        gr.Textbox("This will include the previously made chatbot")
    with gr.Tab("Document Expert"):
        gr.Markdown("# Ask questions about your PDF")

        file_widget = gr.File(file_types=[".pdf"])
        file_widget.upload(rag_instance.process_file, file_widget)

        chatbot = gr.Chatbot(type="messages", height=200)
        user_query = gr.Textbox(placeholder="Type your query here", show_label=False, interactive=True)
        user_query.submit(rag_instance.process_rag_query, [user_query, chatbot], [user_query, chatbot])

demo.launch(server_name="0.0.0.0")
