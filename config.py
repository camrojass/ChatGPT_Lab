#################### Variables ########################

OPENAI_API_KEY = "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"
PINECONE_API_KEY = "4df56ed3-8eea-4e46-adfc-e73f07b665b5"
PINECONE_ENV = "gcp-starter"


# Variables for VectorDataUse
chunk_size=1000
chunk_overlap=200


# Variables for PineconeUse
metric="cosine"
dimension=1536
index_name = "chatgptdbvector"

