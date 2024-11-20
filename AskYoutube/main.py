from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import time
import shutil
import glob
import chromadb
from chromadb.config import Settings
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

load_dotenv()


app = Flask(__name__)
CORS(app)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CHROMA_PATH = "./Chroma"
DATA_PATH = "."  # Directory for Markdown file
aai.settings.api_key = os.getenv("AIAI_API_KEY")

# Global variable for vector store
vector_store = None

# Prompt template for question answering
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def clear_vector_store():
    """Clear the vector store and reset the global variable"""
    global vector_store
    
    try:
        # First, delete the collection and close the client if vector_store exists
        if vector_store is not None:
            try:
                vector_store._client.delete_collection(vector_store._collection.name)
                vector_store._client.reset()
                vector_store = None
                print("[INFO] Cleared vector store collection")
            except Exception as e:
                print(f"[ERROR] Failed to clear vector store collection: {e}")
        
        # Force garbage collection to release file handles
        import gc
        gc.collect()
        
        # Add a small delay to ensure resources are released
        time.sleep(1)
        
        # Now try to remove the directory
        if os.path.exists(CHROMA_PATH):
            try:
                shutil.rmtree(CHROMA_PATH)
                print("[INFO] Cleared existing Chroma database")
            except Exception as e:
                print(f"[WARNING] Failed to clear Chroma database: {e}")
                # If deletion fails, try to rename the directory instead
                try:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    backup_path = f"{CHROMA_PATH}_old_{timestamp}"
                    os.rename(CHROMA_PATH, backup_path)
                    print(f"[INFO] Renamed old Chroma database to {backup_path}")
                except Exception as rename_error:
                    print(f"[ERROR] Failed to rename old Chroma database: {rename_error}")
    
    except Exception as e:
        print(f"[ERROR] Error in clear_vector_store: {e}")
    
    finally:
        # Always recreate the directory
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Reset the vector store variable
        vector_store = None
        
        # Clear system cache
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
            print("[INFO] Cleared system cache")
        except Exception as e:
            print(f"[ERROR] Failed to clear system cache: {e}")

def cleanup_audio_files():
    """Clean up any existing audio and transcription files"""
    # Remove existing MP3 files
    for file in glob.glob("*.mp3"):
        try:
            os.remove(file)
            print(f"[INFO] Removed old audio file: {file}")
        except Exception as e:
            print(f"[ERROR] Failed to remove {file}: {e}")
    
    # Clear the transcription file if it exists
    if os.path.exists("transcription.md"):
        try:
            os.remove("transcription.md")
            print("[INFO] Removed old transcription file")
        except Exception as e:
            print(f"[ERROR] Failed to remove transcription.md: {e}")

def download_and_transcribe_audio(video_link):
    try:
        # Add cleanup at the start of the function
        cleanup_audio_files()
        
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

    with open("transcription.md", "w") as md_file:
        md_file.write(transcription)
    print("[INFO] Transcription saved to transcription.md.")
    return transcription

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

def initialize_chroma(chunks=None):
    global vector_store
    
    try:
        # Initialize Cohere embeddings
        embeddings_model = CohereEmbeddings(
            cohere_api_key=COHERE_API_KEY,
            model="embed-english-v3.0"
        )
        print("[INFO] CohereEmbeddings initialized successfully.")

        # Create a unique collection name using timestamp
        collection_name = f"example_{int(time.time())}"

        # Initialize Chroma vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_model,
            persist_directory=CHROMA_PATH
        )
        
        if chunks:
            print("[INFO] Saving chunks to Chroma.")
            vector_store.add_documents(documents=chunks)
            print(f"[INFO] Saved {len(chunks)} chunks to {CHROMA_PATH}.")
        
        return vector_store

    except Exception as e:
        print(f"[ERROR] Failed to initialize Chroma: {e}")
        raise


def process_and_query_chroma(chunks=None, query_text=None):
    """
    Save chunks to Chroma if needed, and perform a similarity search based on a query.
    """
    global vector_store

    try:
        if vector_store is None:
            initialize_chroma(chunks)

        if query_text:
            print("[INFO] Running main query.")
            try:
                results = vector_store.similarity_search(query_text, k=5)
                print(f"[INFO] Retrieved {len(results)} results from similarity search.")
            except Exception as e:
                print(f"[ERROR] An error occurred during similarity search: {e}")
                return
            
            # Display results
            for res in results:
                print(f"* {res.page_content} [{res.metadata}]")

            # Prepare the context text for prompt generation
            context_text = "\n\n---\n\n".join([res.page_content for res in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            print(f"[INFO] Generated prompt for completion:\n{prompt}")

            # Use Groq for chat completion
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            client = Groq(api_key=api_key)

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )

            response_content = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content
                
            print(f"\n[INFO] Final Response:{response_content}")
            return response_content

    except Exception as e:
        print(f"[ERROR] An error occurred in process_and_query_chroma: {e}")
        raise

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    video_url = data.get('videoUrl')

    try:
        # Clear vector store before processing new video
        clear_vector_store()
        
        # Then proceed with transcription
        transcription = download_and_transcribe_audio(video_url)
        if transcription is None:
            return jsonify({"error": "Failed to transcribe video"}), 500

        # Save transcription to a markdown file
        transcription_path = os.path.join(DATA_PATH, "transcription.md")
        with open(transcription_path, "w") as f:
            f.write(transcription)

        # Load and split documents
        documents = load_documents()
        chunks = split_text(documents)

        # Save the chunks to Chroma
        process_and_query_chroma(chunks=chunks, query_text=None)
        return jsonify({"message": "Transcription completed successfully"})
    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    print("[DEBUG] Received data:", data)  
    
    query_text = data.get('question')
    print("[DEBUG] Extracted query text:", query_text)
    
    try:
        print("[INFO] CohereEmbeddings initialized successfully.")
        # Load and split documents
        documents = load_documents()
        chunks = split_text(documents)

        print("[INFO] Query Text: " + query_text)
        response = process_and_query_chroma(chunks=chunks, query_text=query_text)
        print(f"\n[INFO] Response: {response}")
        
        return jsonify({"response": response}) 
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure Chroma path exists
    os.makedirs(CHROMA_PATH, exist_ok=True)
    app.run(port=5000)