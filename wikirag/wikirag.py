import streamlit as st
import wikipediaapi
import ollama
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Initialize session state for FAISS index
if 'index' not in st.session_state:
    st.session_state.index = None
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to fetch Wikipedia content
def get_wikipedia_content(topic, lang='en'):
    user_agent = "MyWikipediaScraper/1.0 (contact: youremail@mail.com)"
    wiki = wikipediaapi.Wikipedia(language=lang, user_agent=user_agent)
    page = wiki.page(topic)
    return page.text if page.exists() else None

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    return text_splitter.split_text(text)

# Function to create FAISS vector store
def create_vector_store(chunks):
    embedder = st.session_state.embedder
    embeddings = embedder.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    return index, chunks

# Function to retrieve relevant context
def retrieve_context(query, k=3):
    embedder = st.session_state.embedder
    query_embedding = embedder.encode([query])
    
    index = st.session_state.index
    if index is None:
        return None  # No index available yet

    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [st.session_state.documents[i] for i in indices[0]]

# Function to generate answer using DeepSeek (Ollama)
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
        options={'temperature': 0.3, 'max_tokens': 1000}
    )
    return response['response']

# Streamlit UI
st.title("🔍 Wikipedia RAG Q&A with DeepSeek")

# Step 1: Get Wikipedia Content
topic = st.text_input("Enter Wikipedia Topic:")

if st.button("Fetch Content"):
    if topic:
        content = get_wikipedia_content(topic)
        if content:
            chunks = split_text_into_chunks(content)
            index, stored_chunks = create_vector_store(chunks)
            
            # Store FAISS index and documents in session state
            st.session_state.index = index
            st.session_state.documents = stored_chunks
            
            st.success(f"✅ Content for '{topic}' fetched and stored in FAISS!")
            st.text_area("Wikipedia Content (Preview):", content[:1000] + "..." if len(content) > 1000 else content, height=200)
        else:
            st.error(f"❌ The topic '{topic}' does not exist on Wikipedia.")
    else:
        st.warning("⚠ Please enter a topic.")

# Step 2: Ask Questions
if st.session_state.index is not None:
    query = st.text_input("Ask a Question:")
    if st.button("Get Answer"):
        if query:
            context = retrieve_context(query, k=3)
            if context:
                answer = generate_answer_with_ollama(query, context)
                st.subheader("📌 Answer:")
                st.write(answer)
            else:
                st.warning("⚠ No relevant context found.")
        else:
            st.warning("⚠ Please enter a question.")
