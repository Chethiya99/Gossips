import streamlit as st
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
import os  # Removed ddg_images import

# Set your OpenAI key
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

st.set_page_config(page_title="Gossip Genie 💅", layout="centered")
st.title("🧃 Gossip Genie")
st.caption("Ask juicy questions about celebrities mentioned in your gossip files.")

# Upload or load your gossip CSV
uploaded_file = st.file_uploader("Upload your gossip CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load data into LangChain
    loader = DataFrameLoader(df, page_content_column="text")
    data = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(data)

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    db = FAISS.from_documents(docs, embeddings)

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7),
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
