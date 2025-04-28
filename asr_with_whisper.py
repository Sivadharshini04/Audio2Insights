# 📦 Imports
import streamlit as st
import nltk
import whisper
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# 📥 Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# 📥 Load Whisper model once
model = whisper.load_model("small")

# 🎨 Optional: Styling (looks better)
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        font-size: 36px;
        color: #4CAF50;
        text-align: center;
    }
    </style>
    <div class="main">
        <h1 class="title">Audio2Insights</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 🚀 Streamlit App
st.subheader("Transcript and Keyword Analysis of Sample Audio")

# ➡️ Directly load your audio file (no uploading by user)
AUDIO_FILE = "phil_lempert_speech.webm"

if AUDIO_FILE:
    st.success("Using sample audio file: phil_lempert_speech.webm")

    # Transcription
    result = model.transcribe(AUDIO_FILE)
    transcript = result['text']

    st.subheader("Transcript:")
    st.write(transcript)

    # Text Cleaning
    def clean_text(text):
        tokens = word_tokenize(text.lower())
        clean_tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
        return clean_tokens

    tokens = clean_text(transcript)

    # Keyword Extraction
    def extract_keywords(tokens, top_n=10):
        freq_dist = nltk.FreqDist(tokens)
        return freq_dist.most_common(top_n)

    top_keywords = extract_keywords(tokens)
    
    st.subheader("Top Keywords:")
    for word, freq in top_keywords:
        st.write(f"{word}: {freq}")

    # KWIC Viewer
    def kwic_view(text, keyword, window=5):
        words = text.split()
        matches = []
        for i, word in enumerate(words):
            if keyword.lower() in word.lower():
                start = max(i - window, 0)
                end = min(i + window + 1, len(words))
                matches.append(' '.join(words[start:end]))
        return matches

    st.subheader("Keyword-in-Context (KWIC) Search")
    search_keyword = st.text_input("Enter a keyword to search:")

    if search_keyword:
        contexts = kwic_view(transcript, search_keyword)
        if contexts:
            for context in contexts:
                st.write("...", context, "...")
        else:
            st.warning(f"No matches found for '{search_keyword}'.")

