import streamlit as st
import wikipediaapi
import ollama
from docx import Document as DocxDocument

# Function to fetch Wikipedia content
def get_wikipedia_content(topic, lang='en'):
    user_agent = "MyWikipediaScraper/1.0 (contact: youremail@mail.com)"
    wiki = wikipediaapi.Wikipedia(language=lang, user_agent=user_agent)
    page = wiki.page(topic)

    if not page.exists():
        return None
    return page.text

# Function to generate answers using DeepSeek (Ollama)
def generate_answer_with_ollama(query, content):
    prompt = f"""You are an expert assistant trained on document information.
    Use this content to answer the question:
    
    {content}
    
    Question: {query}
    
    Answer in detail using only the provided content:"""
    
    response = ollama.generate(
        model='deepseek-r1:1.5b',
        prompt=prompt,
        options={'temperature': 0.3, 'max_tokens': 2000}
    )
    return response['response']

# Streamlit UI
st.title("📖 Wikipedia Q&A with DeepSeek")

# Step 1: Get Wikipedia Content
topic = st.text_input("Enter Wikipedia Topic:", "")

if st.button("Fetch Content"):
    if topic:
        content = get_wikipedia_content(topic)
        if content:
            st.session_state['content'] = content  # Store content in session state
            st.success(f"✅ Content for '{topic}' fetched successfully!")
            st.text_area("Wikipedia Content:", content[:1000] + "..." if len(content) > 1000 else content, height=200)
        else:
            st.error(f"❌ The topic '{topic}' does not exist on Wikipedia.")
    else:
        st.warning("⚠ Please enter a topic.")

# Step 2: Ask Questions
if 'content' in st.session_state:
    query = st.text_input("Ask a Question:")
    if st.button("Get Answer"):
        if query:
            answer = generate_answer_with_ollama(query, st.session_state['content'])
            st.subheader("📌 Answer:")
            st.write(answer)
        else:
            st.warning("⚠ Please enter a question.")
