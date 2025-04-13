import streamlit as st
import pandas as pd
import os
import re

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from duckduckgo_search import DDGS

# Set Groq API Key
os.environ["GROQ_API_KEY"] = st.secrets["groq"]["api_key"]

# Streamlit UI
st.set_page_config(page_title="Gossip Genie 💅", layout="centered")
st.title("🧃 Gossip Genie")
st.caption("Ask juicy questions about celebrities mentioned in your gossip files.")

# Upload your gossip CSV
uploaded_file = st.file_uploader("Upload your gossip CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load data into LangChain
    loader = DataFrameLoader(df, page_content_column="text")
    data = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(data)

    # Use HuggingFace embeddings (no OpenAI)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    # QA Chain with Groq + LLaMA 3
    qa = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0.7, model_name="llama3-8b-8192"),
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    # Ask questions
    question = st.text_input("🧠 Ask something spicy...")
    if question:
        with st.spinner("Spilling the tea..."):
            result = qa({"query": question})
            st.markdown(f"**🍵 Gossip Answer:** {result['result']}")

            # Show the source gossip
            st.markdown("---")
            st.subheader("📜 Gossip Source")
            for doc in result["source_documents"]:
                st.markdown(f"💬 *{doc.page_content}*")

            # Extract celeb names and show images
            names = re.findall(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', result['result'])
            if names:
                st.markdown("🖼️ **Suspected Celebs:**")
                with DDGS() as ddgs:
                    for name in set(names):
                        st.write(f"🔍 Searching for: {name}")
                        try:
                            image_results = ddgs.images(name + " celebrity", max_results=1)
                            image_results = list(image_results)
                            if image_results:
                                st.image(image_results[0]["image"], caption=name, width=200)
                            else:
                                st.warning(f"No image found for {name}")
                        except Exception as e:
                            st.warning(f"Error fetching image for {name}: {e}")
            else:
                st.info("No celebrity names detected to show images for.")
