# pip install
# langchain
# faiss-cpu (Facebook AI Similarity Search). This library is used to create vector database for similarity dearch
# sentence-transformers
# huggingface-hub
# transformers
# langchain-community


# Document we need to stroe in a place and exist_ok = true will not check if the document exist
# Will create a folder
import os
os.makedirs("data", exist_ok=True)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter

loader = TextLoader("data/india")
docs=loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
len(chunks)

print("Chunks Length :",len(chunks))

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# OLD (to be removed or commented out)
# from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")

db= FAISS.from_documents(chunks, embeddings) # Vector Database

retriever = db.as_retriever()

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=256,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)


def rag_chain(question):
    # 1. retrieve similar docs
    docs = retriever.invoke(question)

    # 2. combine the doc text
    context = "\n\n".join([d.page_content for d in docs])

    # 3. format the prompt
    prompt_text = f"""
                    You are an AI assistant. Use ONLY the provided context to answer.
                    If the answer is NOT in the context, say "Sorry!!!. I don't know".

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """
    # 4. call the model
    raw = llm.invoke(prompt_text)
    answer = raw.replace(prompt_text, "").strip()
    return answer


result = rag_chain("tell about india")
print("Answer ::", result)
    




