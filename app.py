import streamlit as st
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
import os, io, requests, time
# --- Dán đoạn này ngay sau các dòng import ở đầu file ---
from google.generativeai.types import HarmCategory, HarmBlockThreshold

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
# --------------------------------------------------------
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup

# --- CẤU HÌNH ---
st.set_page_config(page_title="Siêu AI Đa Năng", page_icon="🚀", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white;}</style>""", unsafe_allow_html=True)

# --- KẾT NỐI API ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    # Tự động lấy danh sách Model
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

# --- HÀM XỬ LÝ FILE ---
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

def save_docx(content):
    doc = Document()
    for line in content.split('\n'):
        if line.strip(): doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio

def scrape_url(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        return "\n".join([p.get_text() for p in soup.find_all('p')])
    except: return ""

# --- GIAO DIỆN ---
st.title("🚀 Siêu Trợ Lý: Huyền Học - Marketing - Dịch Thuật")

with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)", "🏭 Dịch Thuật Công Nghiệp", "🖼️ Dịch Ảnh (OCR)"])

model = genai.GenerativeModel(selected_model)

# --- 1. HỎI ĐÁP CHUYÊN SÂU ---
if menu == "🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)":
    st.subheader("🔮 Trợ Lý Chuyên Gia (Nạp sách/Dữ liệu)")
    
    with st.sidebar:
        role = st.selectbox("Vai trò AI:", ["Đại sư Huyền học (Giang Công)", "Chuyên gia Marketing & Data", "Trợ lý đa năng"])
        files = st.file_uploader("Nạp tài liệu (PDF/Docx):", accept_multiple_files=True)
        if st.button("Nạp vào bộ não"):
            st.session_state.context = get_text_from_files(files)
            st.success("Đã nạp xong tài liệu!")

    if "context" not in st.session_state: st.session_state.context = ""
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).markdown(m["content"])

    if q := st.chat_input("Hỏi AI..."):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        
        prompt = f"VAI TRÒ: {role}\nKIẾN THỨC BỔ TRỢ: {st.session_state.context}\nCÂU HỎI: {q}"
        
        with st.spinner("AI đang suy nghĩ..."):
            try:
                res = model.generate_content(prompt, safety_settings=safety_settings)
                st.chat_message("assistant").markdown(res.text)
                st.session_state.chat_history.append({"role": "assistant", "content": res.text})
            except Exception as e: st.error(f"Lỗi: {e}")

# --- 2. DỊCH THUẬT CÔNG NGHIỆP ---
elif menu == "🏭 Dịch Thuật Công Nghiệp":
    st.subheader("🏭 Dịch Sách & Truyện Hàng Loạt")
    instr = st.text_area("Yêu cầu dịch (Văn phong, xưng hô...):", value="Dịch sang tiếng Việt mượt mà, dễ hiểu.")
    gloss = st.text_area("Từ điển thuật ngữ:", value="Trúc Cơ, ROI")
    
    tab1, tab2 = st.tabs(["📄 Dịch File", "🌐 Dịch Link Web"])
    
    with tab1:
        up_files = st.file_uploader("Tải nhiều file:", accept_multiple_files=True)
        if st.button("Bắt đầu dịch File"):
            for f in up_files:
                txt = get_text_from_files([f])
                chunks = [txt[i:i+5000] for i in range(0, len(txt), 5000)]
                full_trans = ""
                p_bar = st.progress(0)
                for i, c in enumerate(chunks):
                   # --- BẮT ĐẦU ĐOẠN CODE TỰ ĐỘNG THỬ LẠI ---
import time

# Thử tối đa 3 lần nếu bị lỗi
for attempt in range(3):
    try:
        # Cố gắng gọi AI
        res = model.generate_content(f"YÊU CẦU: {instr}\nTHUẬT NGỮ: {gloss}\nDỊCH ĐOẠN NÀY: {c}", safety_settings=safety_settings)
        break # Nếu thành công (không lỗi) thì thoát vòng lặp ngay
    except Exception as e:
        # Nếu gặp lỗi (bất kể lỗi gì)
        if "ResourceExhausted" in str(e):
            # Nếu là lỗi quá tải, nghỉ 20 giây rồi thử lại
            time.sleep(20) 
        else:
            # Nếu là lỗi khác thì bỏ qua luôn
            break
# --- KẾT THÚC ĐOẠN CODE ---
                    full_trans += res.text + "\n\n"
                    p_bar.progress((i+1)/len(chunks))
                st.download_button(f"Tải bản dịch {f.name}", save_docx(full_trans).getvalue(), f"VN_{f.name}.docx")

    with tab2:
        urls = st.text_area("Dán danh sách Link (mỗi dòng 1 link):")
        if st.button("Bắt đầu dịch Link"):
            links = urls.split("\n")
            all_txt = ""
            for l in links:
                if l.strip():
                    raw = scrape_url(l.strip())
                    res = model.generate_content(f"Dịch nội dung sau: {raw[:15000]}", safety_settings=safety_settings)
                    all_txt += f"\n--- {l} ---\n" + res.text
            st.download_button("Tải file dịch tổng hợp", save_docx(all_txt).getvalue(), "Dich_Web.docx")

# --- 3. DỊCH ẢNH ---
elif menu == "🖼️ Dịch Ảnh (OCR)":
    st.subheader("🖼️ Dịch chữ từ Hình ảnh")
    imgs = st.file_uploader("Tải ảnh lên:", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    if imgs and st.button("Bắt đầu dịch ảnh"):
        full_ocr = ""
        for im_f in imgs:
            img = Image.open(im_f)
            st.image(img, width=300)
            res = model.generate_content(["Nhận diện chữ trong ảnh (kể cả tiếng Trung dọc) và dịch sang Tiếng Việt:", img], safety_settings=safety_settings)
            full_ocr += f"\n--- {im_f.name} ---\n" + res.text
        st.text_area("Kết quả:", full_ocr, height=300)
        st.download_button("Tải file dịch ảnh (.docx)", save_docx(full_ocr).getvalue(), "Dich_Anh.docx")
