# ChatGPT WorkShop
Ejercicios de conexión a ChatGPT vía API usando Python con las librerías Lang Chain y Pinecone.
**URL GitHub:**  https://github.com/camrojass/ChatGPT_Lab.git

## Requerimientos
Python versión 3.10 o superior
* [Python 3.11](https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe)
IDE que soporte python
* [Pycharm CE](https://www.jetbrains.com/es-es/pycharm/download/download-thanks.html?platform=windows&code=PCC)
A continuacion se enlistan las librerías usadas en Python usando ```pip install librery```
* openai
* tiktoken
* langchain
* openai
* chromadb
* langchainhub
* bs4
* pinecone-client

## Parte 1
### Descripción
El código permite enviar mensajes a ChatGPT a través de una API y recuperar las respuestas.

<details><summary>PrompstoChatGpt</summary>
<p>

```python
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
```

</details></p>


## Parte 2
### Descripción
El código permite generar un RAG el cual genera vectores en memoria que se usan para recupera la respuesta a una pregunta echa vía API a ChatGPT.

<details><summary>VectorDataBaseUse</summary>
<p>

```python
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(config.chunk_size, config.chunk_overlap)
splits = text_splitter.split_documents(docs)
print(splits[0])
print(splits[1])

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")

print(response)

```

</details></p>

## Parte 3
### Descripción
El código permite realizar consultas a ChatGPT y almacenar las respuestas utilizando Pinecone, una base de datos vectorial cloud.

<details><summary>PineconeUse</summary>
<p>

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Pinecone
import os
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = config.OPENAI_API_KEY
os.environ["PINECONE_ENV"] = config.PINECONE_ENV

def loadText():
    loader = TextLoader("awedfirstpaper.txt")
    documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    import pinecone

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )



    # First, check if our index already exists. If it doesn't, we create it
    if config.index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(name=config.index_name, metric=config.metric, dimension=config.dimension)
    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=config.index_name)

    query = "What is a distributed pointcut?"

    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)


loadText()
```

</details></p>

### Evidencia
![image](https://github.com/camrojass/ChatGPT_Lab/assets/100396227/fd00d952-be94-46a9-a2cd-c8e0423c39c2)
![image](https://github.com/camrojass/ChatGPT_Lab/assets/100396227/c8ea67e6-fd29-41ca-a666-19e2ce1017e7)


# Autor
* **Luis Daniel Benavides** - *Código inicial* - [dnielben](https://github.com/dnielben) 
* **Camilo Alejandro Rojas** - *Trabajo y documentación* - [camrojass](https://github.com/camrojass)

# Referencias
* OpenAI. Url: https://python.langchain.com/docs/integrations/llms/openai
* Retrieval-augmented generation (RAG). Url: https://python.langchain.com/docs/use_cases/question_answering/
* Pinecone. Url: https://python.langchain.com/docs/integrations/vectorstores/pinecone
