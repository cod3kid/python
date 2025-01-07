from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=api_key)
result = llm("Write a short poem")
print(result)