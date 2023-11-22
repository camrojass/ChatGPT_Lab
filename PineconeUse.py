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
"""
def search():
    embeddings = OpenAIEmbeddings()
    import pinecone

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    query = "What is a distributed pointcut?"

    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)
"""


