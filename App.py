import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the gossip data
@st.cache_data
def load_data():
    df = pd.read_csv("gossips.csv")
    df['embedding'] = df['text'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# App title
st.title("🗣️ Gossip Search App")
st.write("Ask me about any gossip... and I'll spill the tea! ☕")

# User query
query = st.text_input("What do you want to know?")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute similarity
    df['similarity'] = df['embedding'].apply(lambda emb: util.pytorch_cos_sim(query_embedding, emb).item())
    
    # Sort by most similar
    top_matches = df.sort_values('similarity', ascending=False).head(3)

    # Show results
    st.subheader("Top Gossip Snippets 🔍")
    for i, row in top_matches.iterrows():
        st.markdown(f"**•** {row['text']}")

