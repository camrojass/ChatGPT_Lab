from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "¿Cuál es la última fecha de actualización de chatGPT 3.0?"

response = llm_chain.run(question)

print(response)
