import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
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
st.set_page_config(page_title="Ultimate AI Final", page_icon="☯️", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white;}</style>""", unsafe_allow_html=True)

# --- KẾT NỐI API ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key
except:
    st.error("⚠️ Chưa nhập API Key trong Secrets.")
    st.stop()

# --- CÁC HÀM XỬ LÝ (GIỮ NGUYÊN) ---
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

def translate_docx_preserve_layout(file, instruction, glossary):
    doc = Document(file)
    model_trans = genai.GenerativeModel('gemini-1.5-flash')
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
            prompt = f"VAI TRÒ: Biên dịch viên.\nNHIỆM VỤ: Dịch sang Tiếng Việt.\nYÊU CẦU: {instruction}\nTHUẬT NGỮ: {glossary}\nLƯU Ý: Giữ nguyên số lượng đoạn, phân cách bởi [--BREAK--].\n\nVĂN BẢN GỐC:\n{batch_text}"
            try:
                response = model_trans.generate_content(prompt)
                translated_batch = response.text.split("[--BREAK--]")
                for idx, trans_text in zip(current_indices, translated_batch):
                    if idx < len(doc.paragraphs):
                        doc.paragraphs[idx].text = trans_text.strip()
            except: pass
            current_batch = []
            current_indices = []
            bar.progress((i+1)/total_paragraphs)
            time.sleep(1)

    status.text("✅ Đã dịch xong! Ảnh và Bảng biểu được giữ nguyên.")
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

# --- GIAO DIỆN CHÍNH ---
st.title("☯️ Ultimate AI: Đại Sư & Dịch Giả")

menu = st.sidebar.radio("CHỨC NĂNG:", [
    "1. Huấn Luyện & Lưu Trữ (Train Brain)",
    "2. Hỏi Đại Sư (Dùng Bộ Não)",
    "3. Dịch Thuật Đa Năng (Sách/Ảnh/Link)"
])

# --- MODULE 1: HUẤN LUYỆN ---
if menu == "1. Huấn Luyện & Lưu Trữ (Train Brain)":
    st.header("🧠 Huấn Luyện AI")
    st.info("Nạp sách Giang Công, Phong Thủy (PDF/Docx) để tạo 'Bộ Não'.")
    uploaded_files = st.file_uploader("Nạp sách:", accept_multiple_files=True)
    if st.button("Train & Tải Bộ Não"):
        if uploaded_files:
            with st.spinner("Đang học..."):
                raw = get_text_from_files(uploaded_files)
                create_vector_store(get_text_chunks(raw))
                zip_folder("faiss_index_huyenhoc", "bo_nao.zip")
                with open("bo_nao.zip", "rb") as fp:
                    st.download_button("📥 Tải Bộ Não Về", fp, "bo_nao.zip", "application/zip")

# --- MODULE 2: HỎI ĐÁP ---
elif menu == "2. Hỏi Đại Sư (Dùng Bộ Não)":
    st.header("🔮 Hỏi Đáp RAG")
    brain = st.sidebar.file_uploader("Nạp file 'bo_nao.zip':", type="zip")
    vs = None
    if brain:
        with open("temp.zip", "wb") as f: f.write(brain.getbuffer())
        with zipfile.ZipFile("temp.zip", "r") as z: z.extractall(".")
        vs = FAISS.load_local("faiss_index_huyenhoc", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
        st.sidebar.success("Đã nạp não!")
    
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs: st.chat_message(m["role"]).markdown(m["content"])
    
    if q := st.chat_input("Hỏi gì đi..."):
        st.session_state.msgs.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        if vs:
            docs = vs.similarity_search(q, k=4)
            chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-1.5-pro"), chain_type="stuff", prompt=PromptTemplate(template="Dựa vào sách: {context}\nTrả lời: {question}", input_variables=["context", "question"]))
            res = chain({"input_documents": docs, "question": q}, return_only_outputs=True)
            st.session_state.msgs.append({"role": "assistant", "content": res["output_text"]})
            st.chat_message("assistant").markdown(res["output_text"])
        else: st.error("Chưa nạp bộ não!")

# --- MODULE 3: DỊCH THUẬT ĐA NĂNG (ĐẦY ĐỦ 3 TAB) ---
elif menu == "3. Dịch Thuật Đa Năng (Sách/Ảnh/Link)":
    st.header("🏭 Dịch Thuật Công Nghiệp")
    
    col_a, col_b = st.columns(2)
    with col_a:
        instruction = st.text_area("Yêu cầu văn phong:", value="Dịch sang tiếng Việt. Văn phong hay, dễ hiểu. Kiểm tra lỗi chính tả bản gốc trước khi dịch.", height=100)
    with col_b:
        glossary = st.text_area("Từ điển (Glossary):", value="Insight\nROI\nTrúc Cơ", height=100)

    # ĐÂY LÀ PHẦN QUAN TRỌNG BẠN CẦN: 3 TAB
    tab1, tab2, tab3 = st.tabs(["📄 File Word (Giữ Ảnh)", "🌐 Link/Text", "🖼️ Dịch Ảnh (OCR)"])

    # Tab 1: Dịch File Word giữ định dạng
    with tab1:
        st.info("Nạp file Word (.docx). AI sẽ dịch chữ và GIỮ NGUYÊN hình ảnh/bảng biểu.")
        docx_file = st.file_uploader("Tải file Word:", type=['docx'])
        if docx_file and st.button("🚀 Dịch File Word"):
            processed_file = translate_docx_preserve_layout(docx_file, instruction, glossary)
            st.download_button(f"📥 Tải về {docx_file.name}", processed_file.getvalue(), f"VN_{docx_file.name}", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    # Tab 2: Dịch Link hoặc Text
    with tab2:
        st.info("Dán Link truyện hoặc Text. AI sẽ dịch và tạo file Word mới.")
        urls = st.text_area("Dán Link (Mỗi dòng 1 link):")
        if st.button("🚀 Dịch Link"):
            links = urls.split('\n')
            full = ""
            bar = st.progress(0)
            model_t = genai.GenerativeModel('gemini-1.5-flash')
            for i, link in enumerate(links):
                if link.strip():
                    raw = scrape_chapter(link.strip())
                    if raw:
                        try:
                            prompt = f"Yêu cầu: {instruction}\nThuật ngữ: {glossary}\nNội dung: {raw[:15000]}"
                            res = model_t.generate_content(prompt)
                            full += f"\n\n--- {link} ---\n{res.text}"
                        except: pass
                    bar.progress((i+1)/len(links))
            st.download_button("Tải về (.docx)", save_docx_new(full).getvalue(), "Truyen_Web.docx")

    # Tab 3: Dịch Ảnh (OCR) - ĐÃ THÊM LẠI
    with tab3:
        st.info("Tải ảnh chụp sách/truyện (Tiếng Trung/Anh). AI sẽ nhận diện và dịch.")
        uploaded_imgs = st.file_uploader("Tải ảnh:", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        if uploaded_imgs and st.button("🚀 Dịch Ảnh"):
            full_trans = ""
            model_vision = genai.GenerativeModel('gemini-1.5-flash')
            for img_file in uploaded_imgs:
                img = Image.open(img_file)
                st.image(img, width=200, caption=img_file.name)
                
                prompt_vision = f"""
                Bạn là chuyên gia ngôn ngữ.
                1. Nhìn vào ảnh, nhận diện toàn bộ văn bản (kể cả Tiếng Trung phồn/giản, Tiếng Anh).
                2. Dịch sang Tiếng Việt.
                3. YÊU CẦU: {instruction}
                4. THUẬT NGỮ: {glossary}
                """
                try:
                    res = model_vision.generate_content([prompt_vision, img])
                    full_trans += f"\n\n--- Ảnh {img_file.name} ---\n{res.text}"
                except Exception as e:
                    full_trans += f"\n[Lỗi ảnh {img_file.name}: {e}]"
            
            st.text_area("Kết quả:", full_trans, height=300)
            st.download_button("📥 Tải bản dịch Ảnh (.docx)", save_docx_new(full_trans).getvalue(), "Dich_Anh.docx")
