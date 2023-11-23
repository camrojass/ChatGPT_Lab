#################### Variables ########################

OPENAI_API_KEY = "OPENAI_API_KEY"
PINECONE_API_KEY = "PINECONE_API_KEY"
PINECONE_ENV = "gcp-starter"


# Variables for VectorDataUse
chunk_size=1000
chunk_overlap=200


# Variables for PineconeUse
metric="cosine"
dimension=1536
index_name = "chatgptdbvector"

