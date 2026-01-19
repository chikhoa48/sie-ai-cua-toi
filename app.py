import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Siêu AI Đa Năng", page_icon="🚀", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white;}</style>""", unsafe_allow_html=True)

# --- CẤU HÌNH AN TOÀN (MỞ TOANG ĐỂ KHÔNG BỊ CHẶN) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

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
    
    # Ưu tiên Flash vì nó nhanh và ít bị lỗi hạn mức hơn Pro
    if not available_models: 
        available_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
    else:
        # Đảo Flash lên đầu danh sách
        available_models.sort(key=lambda x: "flash" not in x)
except:
    st.error("⚠️ Chưa nhập API Key trong Secrets.")
    st.stop()

# --- CÁC HÀM XỬ LÝ FILE ---
def get_text_from_files(files):
    text = ""
    for f in files:
        try:
            if f.name.endswith('.pdf'):
                reader = PdfReader(f)
                for page in reader.pages: 
                    extracted = page.extract_text()
                    if extracted: text += extracted
            elif f.name.endswith('.docx'):
                doc = Document(f)
                for para in doc.paragraphs: text += para.text + "\n"
            elif f.name.endswith('.txt'):
                text += f.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Lỗi đọc file {f.name}: {e}")
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
        tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        return "\n".join([t.get_text() for t in tags])
    except: return ""

# --- GIAO DIỆN CHÍNH ---
st.title("🚀 Siêu Trợ Lý: Huyền Học - Marketing - Dịch Thuật")

with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.caption("Mẹo: Dùng 'Flash' để dịch nhanh, 'Pro' để thông minh hơn.")
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)", "🏭 Dịch Thuật Công Nghiệp", "🖼️ Dịch Ảnh (OCR)"])

model = genai.GenerativeModel(selected_model)

# ==============================================================================
# 1. HỎI ĐÁP CHUYÊN SÂU
# ==============================================================================
if menu == "🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)":
    st.subheader("🔮 Trợ Lý Chuyên Gia")
    
    with st.sidebar:
        role = st.selectbox("Vai trò AI:", ["Đại sư Huyền học (Giang Công)", "Chuyên gia Marketing & Data", "Trợ lý đa năng"])
        files = st.file_uploader("Nạp tài liệu (PDF/Docx):", accept_multiple_files=True)
        if st.button("Nạp vào bộ não"):
            if files:
                st.session_state.context = get_text_from_files(files)
                st.success(f"Đã nạp xong {len(files)} tài liệu!")
            else:
                st.warning("Chưa chọn file nào!")

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
                if res and res.text:
                    st.chat_message("assistant").markdown(res.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                else:
                    st.error("AI không trả lời được câu này.")
            except Exception as e: st.error(f"Lỗi: {e}")

# ==============================================================================
# 2. DỊCH THUẬT CÔNG NGHIỆP (ĐÃ SỬA LỖI CHI TIẾT)
# ==============================================================================
elif menu == "🏭 Dịch Thuật Công Nghiệp":
    st.subheader("🏭 Dịch Sách & Truyện Hàng Loạt")
    instr = st.text_area("Yêu cầu dịch:", value="Dịch sang tiếng Việt mượt mà, văn phong chuyên nghiệp.")
    gloss = st.text_area("Từ điển thuật ngữ:", value="Trúc Cơ, Nguyên Anh, ROI")
    
    tab1, tab2 = st.tabs(["📄 Dịch File", "🌐 Dịch Link Web"])
    
    # --- TAB DỊCH FILE ---
    with tab1:
        up_files = st.file_uploader("Tải nhiều file:", accept_multiple_files=True)
        
        if st.button("Bắt đầu dịch File"):
            if not up_files:
                st.warning("⚠️ Vui lòng chọn file trước!")
            else:
                for f in up_files:
                    st.info(f"📂 Đang xử lý file: {f.name}...")
                    
                    # 1. Đọc file
                    txt = get_text_from_files([f])
                    
                    # --- KIỂM TRA FILE RỖNG (QUAN TRỌNG) ---
                    if not txt or len(txt.strip()) < 10:
                        st.error(f"❌ File {f.name} không đọc được chữ! (Có thể là file PDF scan/ảnh). Hãy dùng chức năng 'Dịch Ảnh (OCR)' thay thế.")
                        continue
                    # ---------------------------------------

                    chunks = [txt[i:i+20000] for i in range(0, len(txt), 20000)] # Giảm xuống 4000 cho an toàn
                    full_trans = ""
                    p_bar = st.progress(0)
                    
                    st.write(f"👉 File có {len(chunks)} đoạn cần dịch.")

                    # 2. Vòng lặp dịch
                    for i, c in enumerate(chunks):
                        res = None 
                        flag_success = False
                        error_msg = "Chưa rõ nguyên nhân"
                        
                        # Thử lại 3 lần
                        for attempt in range(3):
                            try:
                                prompt_text = f"YÊU CẦU: {instr}\nTHUẬT NGỮ: {gloss}\nNỘI DUNG GỐC:\n{c}"
                                res = model.generate_content(prompt_text, safety_settings=safety_settings)
                                flag_success = True
                                break 
                            except Exception as e:
                                error_msg = str(e)
                                if "ResourceExhausted" in str(e):
                                    st.toast(f"⏳ Mạng bận (Lần {attempt+1}), đợi 20 giây...")
                                    time.sleep(20)
                                else:
                                    time.sleep(2) # Lỗi khác thì đợi ít hơn

                        # Xử lý kết quả
                        if flag_success and res and res.text:
                            full_trans += res.text + "\n\n"
                            st.toast(f"✅ Xong đoạn {i+1}/{len(chunks)}")
                        else:
                            # In lỗi ra màn hình để biết tại sao
                            st.error(f"❌ Lỗi đoạn {i+1}: {error_msg}")
                            if res and res.prompt_feedback:
                                st.caption(f"Chi tiết chặn: {res.prompt_feedback}")
                            
                            full_trans += f"\n[ĐOẠN {i+1} BỊ LỖI: {error_msg}]\n\n"
                        
                        # Cập nhật thanh tiến trình
                        p_bar.progress((i+1)/len(chunks))
                        time.sleep(1) # Nghỉ nhẹ để tránh spam server

                    st.success(f"✅ Hoàn tất file: {f.name}")
                    st.download_button(f"⬇️ Tải về {f.name}", save_docx(full_trans).getvalue(), f"VN_{f.name}.docx")

    # --- TAB DỊCH WEB ---
    with tab2:
        urls = st.text_area("Dán danh sách Link (mỗi dòng 1 link):")
        if st.button("Bắt đầu dịch Link"):
            links = urls.split("\n")
            all_txt = ""
            for l in links:
                if l.strip():
                    raw = scrape_url(l.strip())
                    if raw:
                        try:
                            res = model.generate_content(f"Dịch bài này sang tiếng Việt:\n{raw[:15000]}", safety_settings=safety_settings)
                            if res and res.text:
                                all_txt += f"\n--- {l} ---\n{res.text}\n"
                        except Exception as e:
                            all_txt += f"\n[Lỗi dịch link {l}: {e}]\n"
            st.download_button("Tải file dịch Web", save_docx(all_txt).getvalue(), "Dich_Web.docx")

# ==============================================================================
# 3. DỊCH ẢNH (OCR)
# ==============================================================================
elif menu == "🖼️ Dịch Ảnh (OCR)":
    st.subheader("🖼️ Dịch chữ từ Hình ảnh")
    imgs = st.file_uploader("Tải ảnh lên (PNG/JPG):", accept_multiple_files=True)
    
    if imgs and st.button("Bắt đầu dịch ảnh"):
        full_ocr = ""
        for im_f in imgs:
            try:
                img = Image.open(im_f)
                st.image(img, caption=f"Ảnh: {im_f.name}", width=300)
                
                with st.spinner("Đang soi chữ và dịch..."):
                    res = model.generate_content(
                        ["Trích xuất toàn bộ chữ trong ảnh và dịch sang Tiếng Việt:", img], 
                        safety_settings=safety_settings
                    )
                    if res and res.text:
                        st.write(res.text)
                        full_ocr += f"\n--- {im_f.name} ---\n{res.text}\n"
                    else:
                        st.warning(f"Không đọc được ảnh {im_f.name}")
            except Exception as e:
                st.error(f"Lỗi ảnh {im_f.name}: {e}")
        
        if full_ocr:
            st.download_button("Tải file kết quả", save_docx(full_ocr).getvalue(), "Dich_Anh.docx")
