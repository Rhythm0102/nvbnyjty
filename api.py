import os
import json
import random
import string
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from main import chatbot_response, rewrite_query, improved_get_relevant_chunks, semantic_chunking, tools, populate_pinecone_index, count_tokens
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from embedding_wrapper import EmbeddingWrapper
from dotenv import load_dotenv
import logging
from text_extract import extract_text_from_files
import pickle

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
embedding_wrapper = EmbeddingWrapper(os.getenv('DEEPINFRA_API_KEY'))

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

conversation_history = {}
rewrite_conversation_history = {}
current_index_name = None
docsearch = None
all_text = None
chunks = None

INDEX_FILE = 'current_index.json'
CHUNKS_FILE = 'chunks.pkl'

def load_current_index():
    global current_index_name, docsearch, chunks
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r') as f:
            data = json.load(f)
            current_index_name = data.get('index_name')
            if current_index_name:
                try:
                    index = pc.Index(current_index_name)
                    docsearch = PineconeVectorStore(index, embedding_wrapper, "text")
                    logger.info(f"Loaded existing Pinecone index: {current_index_name}")
                    
                    # Load chunks
                    if os.path.exists(CHUNKS_FILE):
                        with open(CHUNKS_FILE, 'rb') as cf:
                            chunks = pickle.load(cf)
                        logger.info("Loaded existing chunks")
                    else:
                        logger.warning("Chunks file not found")
                except Exception as e:
                    logger.error(f"Error loading existing index or chunks: {str(e)}")
                    current_index_name = None
                    docsearch = None
                    chunks = None

def save_current_index():
    with open(INDEX_FILE, 'w') as f:
        json.dump({'index_name': current_index_name}, f)

def save_chunks():
    with open(CHUNKS_FILE, 'wb') as f:
        pickle.dump(chunks, f)

def generate_index_name():
    return f"hackrx-{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"

def delete_old_index():
    global current_index_name
    if current_index_name:
        try:
            pc.delete_index(current_index_name)
            logger.info(f"Deleted old index: {current_index_name}")
            if os.path.exists(CHUNKS_FILE):
                os.remove(CHUNKS_FILE)
                logger.info("Deleted old chunks file")
        except Exception as e:
            logger.error(f"Error deleting old index or chunks: {str(e)}")

def create_new_index():
    global current_index_name
    current_index_name = generate_index_name()
    pc.create_index(
        current_index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    logger.info(f"Created new Pinecone index: {current_index_name}")
    save_current_index()

@app.route('/upload', methods=['POST'])
def upload_files():
    global docsearch, current_index_name, all_text, chunks

    if 'folder_path' not in request.json:
        return jsonify({"error": "No folder path provided"}), 400

    folder_path = request.json['folder_path']

    if not os.path.isdir(folder_path) and not os.path.isfile(folder_path):
        return jsonify({"error": "Invalid folder or file path"}), 400

    try:
        all_text = extract_text_from_files(folder_path)
        total_tokens = count_tokens(all_text)

        if total_tokens < 6000:
            logger.info("Text is less than 6000 tokens. Bypassing RAG pipeline.")
            current_index_name = None
            docsearch = None
            chunks = None
        else:
            logger.info("Text exceeds 6000 tokens. Proceeding with RAG pipeline.")
            delete_old_index()
            create_new_index()
            docsearch, chunks, _ = populate_pinecone_index(folder_path, embedding_wrapper, pc, current_index_name)
            save_chunks()

        return jsonify({"message": "Files processed successfully.", "index_name": current_index_name}), 200
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return jsonify({"error": f"Error processing files: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global docsearch, all_text, current_index_name, chunks

    data = request.json
    query = data.get('query')
    user_id = request.headers.get('x-user-id')
    session_id = request.headers.get('x-session-id')

    if not query or not user_id or not session_id:
        return jsonify({"error": "Missing required parameters"}), 400

    if session_id not in conversation_history:
        conversation_history[session_id] = []
        rewrite_conversation_history[session_id] = []

    if not docsearch and not all_text:
        load_current_index()
        if not docsearch and not all_text:
            return jsonify({"error": "No index or context available. Please upload files first."}), 400

    if not docsearch:
        response = chatbot_response(
            client,
            os.getenv('MODEL_NAME'),
            query,
            [],
            tools,
            conversation_history[session_id],
            context=all_text
        )
    else:
        rewritten_query, rewrite_conversation_history[session_id] = rewrite_query(
            client,
            os.getenv('MODEL_NAME'),
            query,
            rewrite_conversation_history[session_id]
        )

        if chunks is None:
            return jsonify({"error": "Chunks data is missing. Please upload files again."}), 400

        relevant_chunks = improved_get_relevant_chunks(rewritten_query, docsearch, chunks)

        response = chatbot_response(
            client,
            os.getenv('MODEL_NAME'),
            query,
            relevant_chunks,
            tools,
            conversation_history[session_id]
        )

    conversation_history[session_id].append({"role": "user", "content": query})
    conversation_history[session_id].append({"role": "assistant", "content": response})

    if len(conversation_history[session_id]) > 30:
        conversation_history[session_id] = conversation_history[session_id][-30:]

    return jsonify({"bot_message": response})

if __name__ == '__main__':
    load_current_index()
    app.run(debug=True, port=3000)