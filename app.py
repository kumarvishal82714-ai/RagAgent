from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
#try
# 1. Load data
loader = TextLoader("data.txt")
documents = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector store
db = FAISS.from_documents(chunks, embeddings)

# 5. Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# 6. Ask question
query = " what is array "

# 7. Retrieve relevant docs
docs = db.similarity_search(query, k=3)

context = "\n".join([d.page_content for d in docs])

# 8. Final prompt
prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

# 9. Generate answer
response = llm.invoke(prompt)

print("\n✅ ANSWER:\n")
print(response.content)
