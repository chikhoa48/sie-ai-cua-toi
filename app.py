import streamlit as st
import google.generativeai as genai
import os, io, requests, time, zipfile
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup

# --- KHU VỰC IMPORT LANGCHAIN (FIX LỖI CỰC MẠNH) ---
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
except ImportError:
    st.error("Hệ thống đang khởi tạo thư viện, vui lòng chờ 1-2 phút rồi bấm F5.")
    st.stop()
# -------------------------------------------------------

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Ultimate AI: God Mode", page_icon="☯️", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white;}</style>""", unsafe_allow_html=True)

# --- KẾT NỐI API ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key
    
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                available_models.append(m.name)
    except: pass
    if not available_models: 
        available_models = ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]
except:
    st.error("⚠️ Chưa nhập API Key trong Secrets.")
    st.stop()

# --- CÁC HÀM XỬ LÝ DỮ LIỆU ---
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
    return text_splitter.split_text(text)

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
        return content if len(content) > 100 else soup.get_text()
    except: return ""

def translate_docx_preserve_layout(file, instruction, glossary, model_name):
    doc = Document(file)
    model_trans = genai.GenerativeModel(model_name)
    total_paragraphs = len(doc.paragraphs)
    bar = st.progress(0)
    status = st.empty()
    batch_size = 10
    current_batch, current_indices = [], []
    
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            current_batch.append(text)
            current_indices.append(i)
        if len(current_batch) >= batch_size or (i == total_paragraphs - 1 and current_batch):
            status.text(f"Đang dịch đoạn {i}/{total_paragraphs}...")
            batch_text = "\n[--BREAK--]\n".join(current_batch)
            prompt = f"Dịch sang Tiếng Việt.\nYÊU CẦU: {instruction}\nTHUẬT NGỮ: {glossary}\nLƯU Ý: Giữ nguyên số lượng đoạn, phân cách bởi [--BREAK--].\n\nGỐC:\n{batch_text}"
            try:
                res = model_trans.generate_content(prompt)
                translated = res.text.split("[--BREAK--]")
                for idx, t in zip(current_indices, translated):
                    if idx < len(doc.paragraphs): doc.paragraphs[idx].text = t.strip()
            except: pass
            current_batch, current_indices = [], []
            bar.progress((i+1)/total_paragraphs)
            time.sleep(1)
    status.text("✅ Xong!")
    bio = io.BytesIO()
    doc.save(bio)
    return bio

def save_docx_simple(content):
    doc = Document()
    for line in content.split('\n'):
        if line.strip(): doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio

# --- GIAO DIỆN CHÍNH ---
st.title("☯️ Ultimate AI: God Mode")

with st.sidebar:
    st.header("⚙️ ĐIỀU KHIỂN")
    selected_model = st.selectbox("Chọn Model:", available_models, index=0)
    st.success(f"Dùng: {selected_model}")
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["1. Train Brain (Học Sách)", "2. Hỏi Đại Sư (RAG)", "3. Dịch Thuật Đa Năng"])

if menu == "1. Train Brain (Học Sách)":
    st.header("🧠 Huấn Luyện AI")
    uf = st.file_uploader("Nạp sách (PDF/Docx):", accept_multiple_files=True)
    if st.button("Train & Tải Bộ Não") and uf:
        with st.spinner("Đang học..."):
            raw = get_text_from_files(uf)
            create_vector_store(get_text_chunks(raw))
            zip_folder("faiss_index_huyenhoc", "bo_nao.zip")
            with open("bo_nao.zip", "rb") as fp:
                st.download_button("📥 Tải Bộ Não Về", fp, "bo_nao.zip", "application/zip")

elif menu == "2. Hỏi Đại Sư (RAG)":
    st.header(f"🔮 Luận Giải (Model: {selected_model})")
    brain = st.sidebar.file_uploader("Nạp file 'bo_nao.zip':", type="zip")
    vs = None
    if brain:
        with open("temp.zip", "wb") as f: f.write(brain.getbuffer())
        with zipfile.ZipFile("temp.zip", "r") as z: z.extractall(".")
        vs = FAISS.load_local("faiss_index_huyenhoc", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
        st.sidebar.success("Não đã nạp!")
    
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs: st.chat_message(m["role"]).markdown(m["content"])
    
    if q := st.chat_input("Hỏi Đại sư..."):
        st.session_state.msgs.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        if vs:
            docs = vs.similarity_search(q, k=4)
            chain = load_qa_chain(ChatGoogleGenerativeAI(model=selected_model), chain_type="stuff", prompt=PromptTemplate(template="Dựa vào sách: {context}\nTrả lời: {question}", input_variables=["context", "question"]))
            res = chain({"input_documents": docs, "question": q}, return_only_outputs=True)
            st.session_state.msgs.append({"role": "assistant", "content": res["output_text"]})
            st.chat_message("assistant").markdown(res["output_text"])
        else: st.error("Chưa nạp bộ não!")

elif menu == "3. Dịch Thuật Đa Năng":
    st.header(f"🏭 Dịch Thuật (Động cơ: {selected_model})")
    c1, c2 = st.columns(2)
    with c1: instr = st.text_area("Yêu cầu:", value="Dịch sang tiếng Việt mượt mà.", height=100)
    with c2: gloss = st.text_area("Từ điển:", value="Trúc Cơ\nROI", height=100)
    t1, t2, t3 = st.tabs(["📄 Word (Giữ Ảnh)", "🌐 Link/Text", "🖼️ Dịch Ảnh"])

    with t1:
        df = st.file_uploader("File Word:", type=['docx'])
        if df and st.button("🚀 Dịch File Word"):
            processed = translate_docx_preserve_layout(df, instr, gloss, selected_model)
            st.download_button(f"📥 Tải {df.name}", processed.getvalue(), f"VN_{df.name}", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
    with t2:
        urls = st.text_area("Link truyện:")
        if st.button("🚀 Dịch Link"):
            full = ""
            bar = st.progress(0)
            model_t = genai.GenerativeModel(selected_model)
            url_list = urls.split('\n')
            for i, url in enumerate(url_list):
                if url.strip():
                    raw = scrape_chapter(url.strip())
                    if raw:
                        try:
                            res = model_t.generate_content(f"Yêu cầu: {instr}\nNội dung: {raw[:15000]}")
                            full += f"\n\n--- {url} ---\n{res.text}"
                        except: pass
                    bar.progress((i+1)/len(url_list))
            st.download_button("Tải về", save_docx_simple(full).getvalue(), "Truyen_Dich.docx")
            
    with t3:
        imgs = st.file_uploader("Ảnh:", accept_multiple_files=True, type=['png', 'jpg'])
        if imgs and st.button("🚀 Dịch Ảnh"):
            full_t = ""
            model_v = genai.GenerativeModel(selected_model)
            for img_file in imgs:
                img = Image.open(img_file)
                st.image(img, width=200)
                try:
                    res = model_v.generate_content([f"Dịch sang TV: {instr}", img])
                    full_t += f"\n\n--- {img_file.name} ---\n{res.text}"
                except: pass
            st.text_area("Kết quả:", full_t)
            st.download_button("Tải về", save_docx_simple(full_t).getvalue(), "Dich_Anh.docx")
