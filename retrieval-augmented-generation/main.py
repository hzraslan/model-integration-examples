from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import os

os.environ["GROQ_API_KEY"] = "your-api-key"

llm = Groq(model="llama-3.2-1b-preview")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
model_card = SimpleDirectoryReader(input_files=["o1_system_card.pdf"]).load_data()

index = VectorStoreIndex.from_documents(model_card)

query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("What are the Preparedness Framework risk categories")
print(response)


# messages = [
#     ChatMessage(
#         role="system", content="You are a sarcastic comedian who likes to make technology jokes"
#     ),
#     ChatMessage(role="user", content="What is USB C?"),
# ]
#
# response = llm.stream_chat(messages)
#
# for chunk in response:
#     print(chunk.delta, end="")