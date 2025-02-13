# 1. Install required packages
# pip install pypdf langchain sentence-transformers faiss-cpu ollama

import ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Add this import
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 2. Load and process PDF document
def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()  # Returns list of Document objects

# 3. Split text into chunks
def split_documents(pages):  # Changed parameter name
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(pages)  # Directly split Document objects

# 4. Create vector store
def create_vector_store(split_docs):  # Changed parameter name
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    document_texts = [doc.page_content for doc in split_docs]  # Use .page_content
    embeddings = embedder.encode(document_texts)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index, document_texts, embedder

# 5. Retrieve relevant context (unchanged)
def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]

# 6. Generate answer using Ollama (unchanged)
def generate_answer_with_ollama(query, context):
    formatted_context = "\n".join(context)
    
    prompt = f"""You are an expert assistant trained on document information.
    Use this context to answer the question:
    
    {formatted_context}
    
    Question: {query}
    
    Answer in detail using only the provided context:"""
    
    response = ollama.generate(
        model='deepseek-r1:1.5b',
        prompt=prompt,
        options={
            'temperature': 0.3,
            'max_tokens': 2000
        }
    )
    return response['response']

# Main workflow (modified)
def main(pdf_path, query):
    # Load and process PDF
    pages = load_pdf_documents(pdf_path)  # Get Document objects
    split_docs = split_documents(pages)  # Split properly
    
    # Create vector store
    index, document_texts, embedder = create_vector_store(split_docs)
    
    # Retrieve context
    context = retrieve_context(query, embedder, index, document_texts)
    
    # Generate answer
    answer = generate_answer_with_ollama(query, context)
    return answer

# Example usage (unchanged)
if __name__ == "__main__":
    pdf_path = "President.pdf"
    query = "Who is current president of usa ?"
    
    result = main(pdf_path, query)
    print("Answer:", result)