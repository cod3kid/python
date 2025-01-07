from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=api_key)

code_prompt = PromptTemplate(
    template="Write a short {language} function that will {task}",
    input_variables=["language","task"]
    )

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

result = code_chain({
    "language":"python",
    "task":"return a list of numbers"
})

print(result)