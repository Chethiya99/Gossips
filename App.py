import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load model using CPU only to avoid deployment issues
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

model = load_model()

# Load gossip data
@st.cache_data
def load_data():
    df = pd.read_csv("gossips.csv")
    df['embedding'] = df['text'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# Streamlit App UI
st.title("🗣️ Gossip AI Search")
st.write("Ask me about any gossip and I’ll give you the scoop! ☕")

# Text input
query = st.text_input("Enter your gossip question here...")

# Search logic
if query:
    with st.spinner("Looking through all the juicy stories..."):
        query_embedding = model.encode(query, convert_to_tensor=True)
        df['similarity'] = df['embedding'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())
        top_matches = df.sort_values('similarity', ascending=False).head(3)

        st.subheader("👀 Here's what I found:")
        for i, row in top_matches.iterrows():
            st.markdown(f"**•** {row['text']}")
