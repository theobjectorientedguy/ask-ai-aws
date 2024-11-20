from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
import pickle
import time
import shutil
import glob
import stat
import assemblyai as aai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import ChatPromptTemplate
from pytubefix import YouTube
from pytubefix.cli import on_progress
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

# Initialize the Flask app
app = Flask(__name__)

# API keys and paths (Removed sensitive keys)
# Load environment variables
load_dotenv()

# API keys and paths
COHERE_API_KEY = os.getenv("BjJwK9jegTX0DzKIHK3LD44Q9yfBTrnz4qO453la")  # Use environment variables for API keys
CHROMA_PATH = "chroma"
DATA_PATH = "."
aai.settings.api_key = os.getenv("2a09277c23684af9b374eea68ed5613a")  # Set AssemblyAI API key from environment

# Prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def download_and_transcribe_audio(video_link):
    try:
        print(f"[INFO] Starting download for video: {video_link}")
        yt = YouTube(video_link, on_progress_callback=on_progress)
        print(f"[INFO] Video title: {yt.title}")

        audio_stream = yt.streams.get_audio_only()
        audio_stream.download(mp3=True)
        print("[INFO] Audio downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download audio: {e}")
        return None

    audio_files = glob.glob("*.mp3")
    if audio_files:
        audio_file_path = audio_files[0]
        print(f"[INFO] Found audio file: {audio_file_path}")
    else:
        print("[ERROR] No .mp3 file found in the current directory.")
        return None

    # Transcribe audio
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file_path)
        transcription = transcript.text
        print("[INFO] Transcription completed.")
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None

    transcription_lines = transcription.split("\n")
    with open("transcription.md", "w") as md_file:
        for line in transcription_lines:
            md_file.write(line + "\n")
    print("[INFO] Transcription saved to transcription.md.")
    return transcription_lines

def load_documents():
    print("[INFO] Loading documents from directory.")
    markdown_path = os.path.join(DATA_PATH, "transcription.md")
    loader = UnstructuredMarkdownLoader(markdown_path)
    document = loader.load()
    print(f"[INFO] Loaded {len(document)} documents.")
    return document

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Generated {len(chunks)} chunks.")
    return chunks

def process_and_query_chroma(chunks=None, query_text=None):
    # Initialize Cohere embeddings
    embeddings_model = CohereEmbeddings(
        cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0"
    )
    print("[INFO] CohereEmbeddings initialized successfully.")
    
    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.chmod(CHROMA_PATH, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # Initialize Chroma vector store
    vector_store = Chroma(
        collection_name="example",
        embedding_function=embeddings_model,
        persist_directory="./Chroma",  # Where to save data locally
    )

    # Step 1: Save to Chroma if chunks are provided
    if chunks is not None:
        print("[INFO] Saving chunks to Chroma.")
        try:
            vector_store.add_documents(documents=chunks)
            vector_store.persist()
            print(f"[INFO] Saved {len(chunks)} chunks to {CHROMA_PATH}.")
        except Exception as e:
            print(f"[ERROR] Error adding chunks to Chroma: {e}")
            return []

    # Step 2: Query Chroma if a query_text is provided
    results = []
    if query_text:
        print("[INFO] Running main query.")
        try:
            results = vector_store.similarity_search(query_text, k=5)
            print(f"[INFO] Retrieved {len(results)} results from similarity search.")
        except Exception as e:
            print(f"[ERROR] An error occurred during similarity search: {e}")

    return results

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    transcription = download_and_transcribe_audio(video_url)
    if transcription:
        return jsonify({"transcription": "\n".join(transcription)})
    else:
        return jsonify({"error": "Failed to transcribe video"}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query_text")
    if not query_text:
        return jsonify({"error": "No query text provided"}), 400

    documents = load_documents()
    chunks = split_text(documents)
    results = process_and_query_chroma(chunks=chunks, query_text=query_text)

    # Prepare results for frontend
    response = {
        "results": [{"content": res.page_content, "source": res.metadata.get("source", "")} for res in results]
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
