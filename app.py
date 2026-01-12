import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import io
import requests
from bs4 import BeautifulSoup
import time

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Siêu AI Đa Năng (All-in-One)", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stButton>button {background-color: #2e86de; color: white;}
    .main {background-color: #f1f2f6;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Siêu AI: Marketing - Tài Chính - Dịch Thuật")

# --- KẾT NỐI API ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("⚠️ Chưa nhập API Key. Hãy vào Settings -> Secrets để nhập nhé.")
    st.stop()

# --- CÁC HÀM XỬ LÝ (FUNCTION) ---
def get_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages: text += page.extract_text() or ""
    return text

def get_docx_text(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs: text += para.text + "\n"
    return text

def get_csv_txt_text(file):
    stringio = io.StringIO(file.getvalue().decode("utf-8"))
    return stringio.read()

def extract_text_from_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return "\n\n".join([p.get_text() for p in paragraphs])
    except Exception as e:
        return f"Lỗi đọc web: {e}"

def save_to_docx(translated_text):
    doc = Document()
    doc.add_heading('Bản Dịch Bởi AI', 0)
    for line in translated_text.split('\n'):
        if line.strip(): doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio

# --- SIDEBAR: MENU CHỨC NĂNG ---
with st.sidebar:
    st.header("🎛️ MENU CHỨC NĂNG")
    
    # CHỌN CHẾ ĐỘ
    app_mode = st.radio(
        "Bạn muốn dùng tính năng gì?",
        [
            "1. Chat & Phân Tích (Marketing/Tài Chính)", 
            "2. Dịch Sách & Truyện (Batch/URL)", 
            "3. Dịch Ảnh (OCR)"
        ]
    )
    st.divider()

# --- KHU VỰC 1: CHAT & PHÂN TÍCH (MARKETING, TÀI CHÍNH, PHONG THỦY) ---
if app_mode == "1. Chat & Phân Tích (Marketing/Tài Chính)":
    st.subheader("💬 Trợ Lý Chuyên Gia & Phân Tích Dữ Liệu")
    
    # Cấu hình bên trái
    with st.sidebar:
        role = st.selectbox("Chọn vai trò AI:", [
            "Chuyên Gia Marketing (Content/Insight)", 
            "Chuyên Gia Tài Chính (ROI/Đầu tư)",
            "Thầy Phong Thủy (Giang Công)",
            "Trợ Lý Bình Thường"
        ])
        
        uploaded_files = st.file_uploader("Nạp dữ liệu (PDF, Word, CSV, TXT):", accept_multiple_files=True)
        if st.button("🔄 Nạp dữ liệu"):
            with st.spinner("Đang đọc..."):
                raw = ""
                for f in uploaded_files:
                    if f.name.endswith('.pdf'): raw += get_pdf_text(f)
                    elif f.name.endswith('.docx'): raw += get_docx_text(f)
                    elif f.name.endswith('.csv') or f.name.endswith('.txt'): raw += get_csv_txt_text(f)
                st.session_state.context_text = raw
                st.session_state.messages = [] # Reset chat
                st.success("Đã nạp xong!")

    # Logic Chat
    if "messages" not in st.session_state: st.session_state.messages = []
    if "context_text" not in st.session_state: st.session_state.context_text = ""

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Hỏi gì đi (Ví dụ: Viết content, Phân tích ROI...):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Prompt engineering cho từng vai
        instructions = ""
        if "Marketing" in role: instructions = "Bạn là CMO thực chiến. Hãy viết content thu hút, phân tích insight sâu sắc, tìm nỗi đau khách hàng."
        elif "Tài Chính" in role: instructions = "Bạn là chuyên gia tài chính CFA. Tập trung vào số liệu, ROI, dòng tiền và rủi ro."
        elif "Phong Thủy" in role: instructions = "Bạn là thầy Phong Thủy phái Giang Công. Dùng từ ngữ trang trọng, cổ học."

        full_prompt = f"{instructions}\n\nDựa vào dữ liệu: {st.session_state.context_text}\n\nCâu hỏi: {prompt}"
        
        try:
            res = model.generate_content(full_prompt)
            with st.chat_message("assistant"): st.markdown(res.text)
            st.session_state.messages.append({"role": "assistant", "content": res.text})
        except Exception as e: st.error(f"Lỗi chi tiết: {e}")

# --- KHU VỰC 2: DỊCH SÁCH & TRUYỆN (BATCH MODE) ---
elif app_mode == "2. Dịch Sách & Truyện (Batch/URL)":
    st.subheader("📚 Cỗ Máy Dịch Thuật: Truyện & Sách Chuyên Ngành")
    
    with st.sidebar:
        st.info("Cấu hình Dịch Thuật")
        trans_source = st.radio("Nguồn:", ["File Tài Liệu (PDF/Docx/Txt)", "Link Website"])
        glossary = st.text_area("Từ điển thuật ngữ (Giữ nguyên từ):", "Ví dụ:\nTrúc Cơ\nROI\nInsight", height=100)
    
    if trans_source == "File Tài Liệu (PDF/Docx/Txt)":
        ufile = st.file_uploader("Tải sách lên:", type=['txt', 'docx', 'pdf'])
        if ufile and st.button("🚀 Bắt đầu Dịch File"):
            # Đọc file
            raw_text = ""
            if ufile.name.endswith('.pdf'): raw_text = get_pdf_text(ufile)
            elif ufile.name.endswith('.docx'): raw_text = get_docx_text(ufile)
            elif ufile.name.endswith('.txt'): raw_text = get_csv_txt_text(ufile)
            
            # Cắt nhỏ và dịch
            chunk_size = 3000
            chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
            
            full_trans = ""
            my_bar = st.progress(0)
            status = st.empty()
            
            for i, chunk in enumerate(chunks):
                status.text(f"Đang dịch phần {i+1}/{len(chunks)}...")
                p = f"Dịch đoạn sau sang Tiếng Việt. Văn phong trôi chảy. Thuật ngữ bắt buộc giữ: {glossary}\n\nNội dung:\n{chunk}"
                try:
                    r = model.generate_content(p)
                    full_trans += r.text + "\n\n"
                    my_bar.progress((i+1)/len(chunks))
                    time.sleep(1)
                except: full_trans += f"[Lỗi đoạn {i+1}]"
            
            status.text("✅ Xong!")
            st.text_area("Kết quả:", full_trans, height=200)
            
            # Tải về
            docx = save_to_docx(full_trans)
            st.download_button("📥 Tải bản dịch (.docx)", docx.getvalue(), "Ban_dich.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    else: # Dịch Link Web
        url = st.text_input("Dán Link chương truyện:")
        if url and st.button("🚀 Dịch Chương Này"):
            with st.spinner("Đang cào và dịch..."):
                content = extract_text_from_url(url)
                if len(content) > 50:
                    p = f"Dịch truyện sau sang Tiếng Việt. Văn phong cuốn hút. Thuật ngữ giữ nguyên: {glossary}\n\nNội dung:\n{content[:15000]}"
                    res = model.generate_content(p)
                    st.markdown(res.text)
                    docx = save_to_docx(res.text)
                    st.download_button("📥 Tải về (.docx)", docx.getvalue(), "Chuong_truyen.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                else: st.error("Không đọc được web này.")

# --- KHU VỰC 3: DỊCH ẢNH (OCR) ---
elif app_mode == "3. Dịch Ảnh (OCR)":
    st.subheader("🌏 Dịch Thuật Hình Ảnh (Anh/Hoa -> Việt)")
    img = st.file_uploader("Tải ảnh lên:", type=["jpg", "png"])
    
    if img:
        image = Image.open(img)
        st.image(image, caption="Ảnh gốc", width=400)
        if st.button("🚀 Dịch Ngay"):
            with st.spinner("AI đang nhìn và dịch..."):
                p = "Dịch toàn bộ chữ trong ảnh sang Tiếng Việt. Văn phong tự nhiên. Nếu là sách chuyên ngành hãy giữ thuật ngữ."
                res = model.generate_content([p, image])
                st.markdown("### Kết quả:")
                st.write(res.text)
