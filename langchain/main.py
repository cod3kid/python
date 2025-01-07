from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task",default="return a list of numbers")
parser.add_argument("--language",default="python")
args = parser.parse_args()



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
    "language":args.language,
    "task": args.task
})

print(result["text"])