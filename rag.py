import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
# Step 1: Load documents (example corpus)
documents = [
    "Python is a popular programming language for AI.",
    "Threading in Python allows concurrent execution of code.",
    "Llama 3.2 is a state-of-the-art language model by Ollama.",
    "RAG stands for Retrieval-Augmented Generation.",
  
]

# Step 2: Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# Function to retrieve relevant documents
def retrieve_docs(query):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)
    most_similar_idx = similarities.argmax()
    return documents[most_similar_idx]

# Step 3: Query the RAG system
def rag_query(query):
    relevant_doc = retrieve_docs(query)
    print(f"Retrieved Document: {relevant_doc}")
    
    # Step 4: Send the relevant document and query to Ollama API (using Llama 3.2)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": f"{relevant_doc}\n\nQuery: {query}"}
    )
   
      # Handle the raw response text
        # Handle the raw response text
    raw_response = response.text
     # Extract all 'response' values from the JSON objects
    responses = re.findall(r'"response":\s*"([^"]+)"', raw_response)
    # Combine all parts into a complete sentence
    final_response = "".join(responses)
    print(f"Llama Response: {final_response}")

# Example Query
rag_query("What is RAG?")
