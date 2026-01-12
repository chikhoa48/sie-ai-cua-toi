import streamlit as st
import google.generativeai as genai

# --- KHU VỰC SỬA LỖI IMPORT (QUAN TRỌNG) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# FIX LỖI 1: Dùng gạch dưới _ thay vì dấu chấm .
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# FIX LỖI 2: Đường dẫn đầy đủ cho load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# -------------------------------------------

from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import io
import requests
from bs4 import BeautifulSoup
import time
import zipfile
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="Ultimate AI: God Mode (Fixed v2)", page_icon="☯️", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white;}</style>""", unsafe_allow_html=True)

# --- KẾT NỐI API ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Quét model
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                available_models.append(m.name)
    except: pass
    if not available_models: available_models = ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]
except:
    st.error("⚠️ Chưa nhập API Key.")
    st.stop()

# --- CÁC HÀM XỬ LÝ ---
def get_text_from_files(files):
    text = ""
    for f in files:
        if f.name.endswith('.pdf'):
            reader = PdfReader(f)
            for page in reader.pages: text += page.extract_text() or ""
        elif f.name.endswith('.docx'):
            doc = Document(f)
            for para in doc.paragraphs: text += para.text + "\n"
        elif f.name.endswith('.txt'):
            text += f.getvalue().decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_huyenhoc")
    return vector_store

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(folder_path, '..')))

def scrape_chapter(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        content = "\n".join([p.get_text() for p in soup.find_all('p')])
        if len(content) < 100: content = soup.get_text()
        return content
    except: return ""

def translate_docx_preserve_layout(file, instruction, glossary, model_name):
    doc = Document(file)
    model_trans = genai.GenerativeModel(model_name)
    total_paragraphs = len(doc.paragraphs)
    bar = st.progress(0)
    status = st.empty()
    batch_size = 10
    current_batch = []
    current_indices = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            current_batch.append(text)
            current_indices.append(i)
        if len(current_batch) >= batch_size or (i == total_paragraphs - 1 and current_batch):
            status.text(f"Đang dịch đoạn {i}/{total_paragraphs}...")
            batch_text = "\n[--BREAK--]\n".join(current_batch)
            prompt = f"VAI TRÒ: Dịch giả.\nYÊU CẦU: {instruction}\nTHUẬT NGỮ: {glossary}\nLƯU Ý: Giữ nguyên số lượng đoạn, phân cách bởi [--BREAK--].\n\nGỐC:\n{batch_text}"
            try:
                res = model_trans.generate_content(prompt)
                translated = res.text.split("[--BREAK--]")
                for idx, t in zip(current_indices, translated):
                    if idx < len(doc.paragraphs): doc.paragraphs[idx].text = t.strip()
            except: pass
            current_batch = []
            current_indices = []
            bar.progress((i+1)/total_paragraphs)
            time.sleep(1)
    status.text("✅ Xong!")
    bio = io.BytesIO()
    doc.save(bio)
    return bio

def save_docx_new(content):
    doc = Document()
    for line in content.split('\n'):
        if line.strip(): doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio

# --- GIAO DIỆN ---
st.title("☯️ Ultimate AI: God Mod
