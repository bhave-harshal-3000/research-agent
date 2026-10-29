import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # Loads variables from .env into the environment

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

response = llm.invoke("Explain an AI research agent in one sentence.")
print(response.content)