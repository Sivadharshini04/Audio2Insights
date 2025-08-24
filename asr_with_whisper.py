import streamlit as st
import whisper
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import atexit
import tempfile
from pytube import YouTube


nltk.download('punkt')
nltk.download('stopwords')

model = whisper.load_model("small")


@atexit.register
def cleanup_temp():
    try:
        if os.path.exists("downloaded_audio.mp3"):
            os.remove("downloaded_audio.mp3")
    except Exception:
        pass

st.title("Audio2Insights 🔍")
st.write("Transcribe & analyze audio from file or YouTube.")

option = st.radio("Input Method:", ["Upload Audio File", "YouTube Link"])

transcript = ""
import yt_dlp
import uuid
FFMPEG_PATH = r"C:\ffmpeg-7.1.1-essentials_build\bin"  # update if different

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload (.mp3/.wav/.webm)", type=["mp3", "wav", "webm"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name
        st.success("Transcribing...")
        result = model.transcribe(temp_path)
        transcript = result['text']


elif option == "YouTube Link":
    yt_link = st.text_input("Paste YouTube Link")
    if yt_link:
        try:
            if os.path.exists("downloaded_audio.mp3"):
                os.remove("downloaded_audio.mp3")

            unique_name = f"yt_audio_{uuid.uuid4().hex}.mp3"

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'ffmpeg_location': r"C:\ffmpeg-7.1.1-essentials_build\bin",
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_link])

            for f in os.listdir():
                if f.startswith("temp_audio") and f.endswith(".mp3"):
                    os.rename(f, unique_name)

            st.success("Audio downloaded. Transcribing...")
            result = model.transcribe(unique_name)
            transcript = result['text']

            os.remove(unique_name)

        except Exception as e:
            st.error(f"Error downloading: {e}")


if transcript:
    st.subheader("Transcript")
    st.write(transcript)

    def clean_text(text):
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w.isalpha() and w not in stopwords.words("english")]

    tokens = clean_text(transcript)

    def extract_keywords(tokens, top_n=10):
        return nltk.FreqDist(tokens).most_common(top_n)

    top_keywords = extract_keywords(tokens)
    st.subheader("Top Keywords")
    for word, freq in top_keywords:
        st.write(f"{word}: {freq}")

    def kwic_view(text, keyword, window=5):
        words = text.split()
        contexts = []
        for i, word in enumerate(words):
            if keyword.lower() in word.lower():
                start = max(i - window, 0)
                end = min(i + window + 1, len(words))
                contexts.append("... " + " ".join(words[start:end]) + " ...")
        return contexts

    st.subheader("KWIC Search")
    keyword = st.text_input("Enter keyword:")
    if keyword:
        results = kwic_view(transcript, keyword)
        if results:
            for r in results:
                st.write(r)
        else:
            st.warning("No matches found.")
