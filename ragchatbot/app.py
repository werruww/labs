import streamlit as st
import ollama
import faiss
import numpy as np
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Function to load and process PDF
def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Function to split text into chunks
def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(pages)

# Function to create vector store
def create_vector_store(split_docs):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    document_texts = [doc.page_content for doc in split_docs]
    embeddings = embedder.encode(document_texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    return index, document_texts, embedder

# Function to retrieve relevant context
def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]

# Function to generate answer using Ollama
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
        options={'temperature': 0.3, 'max_tokens': 2000}
    )
    return response['response']

# Function to simulate a smooth typing effect
def typing_effect(text, delay=0.05):
    typed_text = ""  # Store final output
    placeholder = st.empty()  # Create a placeholder

    for char in text:
        typed_text += char  # Accumulate text
        placeholder.markdown(f"**Answer:** {typed_text}")  # Update output
        time.sleep(delay)  # Simulate typing speed

# Streamlit App
st.title("📄 AI Chatbot for PDF Documents")
st.sidebar.header("Upload a PDF")

uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        pdf_path = f"./uploaded_file.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the PDF
        pages = load_pdf_documents(pdf_path)
        split_docs = split_documents(pages)
        index, document_texts, embedder = create_vector_store(split_docs)
        st.session_state["index"] = index
        st.session_state["documents"] = document_texts
        st.session_state["embedder"] = embedder
        st.success("✅ PDF uploaded and processed successfully!")

# Chat interface
st.subheader("💬 Ask Questions from the PDF")

# Disable text input while processing
text_disabled = st.session_state.get("processing", False)
query = st.text_input("Ask your question:", disabled=text_disabled)

if st.button("Get Answer"):
    if query:
        st.session_state["processing"] = True  # Disable input

        with st.spinner("🤖 Thinking..."):
            context = retrieve_context(query, st.session_state["embedder"], st.session_state["index"], st.session_state["documents"])
            answer = generate_answer_with_ollama(query, context)
                
        # Simulate typing effect in a single output
        typing_effect(answer)
        st.session_state["processing"] = False  # Re-enable input

    else:
        st.warning("⚠️ Please enter a question.")

st.sidebar.write("Built with ❤️ using Streamlit and Ollama")
