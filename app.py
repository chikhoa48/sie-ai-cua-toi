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
# Dùng BLOCK_NONE để AI chấp nhận dịch mọi nội dung
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- KẾT NỐI API ---
try:
    # Lấy API Key từ Secrets của Streamlit
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    # Tự động lấy danh sách Model
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                available_models.append(m.name)
    except: pass
    # Nếu không lấy được thì dùng mặc định
    if not available_models: 
        available_models = ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]
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
            st.error(f"Lỗi khi đọc file {f.name}: {e}")
    return text

def save_docx(content):
    doc = Document()
    for line in content.split('\n'):
        # Loại bỏ các dòng trống thừa
        if line.strip(): doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio

def scrape_url(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        # Lấy text từ các thẻ p, h1, h2...
        tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        return "\n".join([t.get_text() for t in tags])
    except: return ""

# --- GIAO DIỆN CHÍNH ---
st.title("🚀 Siêu Trợ Lý: Huyền Học - Marketing - Dịch Thuật")

with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)", "🏭 Dịch Thuật Công Nghiệp", "🖼️ Dịch Ảnh (OCR)"])

# Khởi tạo model
model = genai.GenerativeModel(selected_model)

# ==============================================================================
# 1. HỎI ĐÁP CHUYÊN SÂU
# ==============================================================================
if menu == "🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)":
    st.subheader("🔮 Trợ Lý Chuyên Gia (Nạp sách/Dữ liệu)")
    
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

    # Hiện lịch sử chat
    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).markdown(m["content"])

    # Xử lý câu hỏi
    if q := st.chat_input("Hỏi AI..."):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        
        prompt = f"VAI TRÒ: {role}\nKIẾN THỨC BỔ TRỢ TỪ FILE: {st.session_state.context}\nCÂU HỎI: {q}"
        
        with st.spinner("AI đang suy nghĩ..."):
            try:
                res = model.generate_content(prompt, safety_settings=safety_settings)
                if res and res.text:
                    st.chat_message("assistant").markdown(res.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                else:
                    st.error("AI không trả lời (Có thể do nội dung bị chặn).")
            except Exception as e: st.error(f"Lỗi: {e}")

# ==============================================================================
# 2. DỊCH THUẬT CÔNG NGHIỆP (PHẦN BẠN CẦN SỬA NHIỀU NHẤT)
# ==============================================================================
elif menu == "🏭 Dịch Thuật Công Nghiệp":
    st.subheader("🏭 Dịch Sách & Truyện Hàng Loạt")
    instr = st.text_area("Yêu cầu dịch (Văn phong, xưng hô...):", value="Dịch sang tiếng Việt mượt mà, văn phong kiếm hiệp/ngôn tình. Giữ nguyên các thuật ngữ Hán Việt quan trọng.")
    gloss = st.text_area("Từ điển thuật ngữ (AI sẽ ưu tiên dùng):", value="Trúc Cơ, Nguyên Anh, ROI, Marketing")
    
    tab1, tab2 = st.tabs(["📄 Dịch File", "🌐 Dịch Link Web"])
    
    # --- TAB DỊCH FILE ---
    with tab1:
        up_files = st.file_uploader("Tải nhiều file:", accept_multiple_files=True)
        
        if st.button("Bắt đầu dịch File"):
            if not up_files:
                st.warning("Vui lòng chọn file trước!")
            else:
                for f in up_files:
                    st.write(f"⏳ Đang xử lý file: **{f.name}**...")
                    
                    # 1. Đọc nội dung file
                    txt = get_text_from_files([f])
                    if not txt:
                        st.warning(f"File {f.name} rỗng hoặc không đọc được.")
                        continue
                        
                    # 2. Cắt nhỏ văn bản (Mỗi đoạn 5000 ký tự)
                    chunks = [txt[i:i+5000] for i in range(0, len(txt), 5000)]
                    full_trans = ""
                    p_bar = st.progress(0) # Thanh tiến trình
                    
                    # 3. Duyệt qua từng đoạn để dịch
                    for i, c in enumerate(chunks):
                        res = None 
                        flag_success = False
                        
                        # --- CƠ CHẾ THỬ LẠI (RETRY LOGIC) ---
                        for attempt in range(3):
                            try:
                                # Tạo prompt
                                prompt_text = f"YÊU CẦU: {instr}\nTHUẬT NGỮ: {gloss}\nNỘI DUNG CẦN DỊCH:\n{c}"
                                
                                # Gọi AI
                                res = model.generate_content(prompt_text, safety_settings=safety_settings)
                                flag_success = True
                                break # Thành công thì thoát vòng lặp thử lại
                                
                            except Exception as e:
                                # Nếu gặp lỗi ResourceExhausted (Hết hạn mức/Quá nhanh)
                                if "ResourceExhausted" in str(e):
                                    if attempt < 2:
                                        st.toast(f"Mạng bận, đang thử lại đoạn {i+1} (Lần {attempt+1})...")
                                        time.sleep(20) # Nghỉ 20 giây
                                    else:
                                        st.error(f"❌ Dừng lại ở đoạn {i+1} do Google chặn quá tải.")
                                else:
                                    st.caption(f"Lỗi lạ ở đoạn {i+1}: {e}")
                                    break
                        
                        # --- XỬ LÝ KẾT QUẢ ---
                        if flag_success and res and res.text:
                            full_trans += res.text + "\n\n"
                        else:
                            full_trans += f"\n[Đoạn {i+1} bị lỗi hoặc AI từ chối dịch]\n\n"
                        
                        # Cập nhật thanh tiến trình (Dòng này đã được căn lề chuẩn)
                        p_bar.progress((i+1)/len(chunks))
                        time.sleep(2) # Nghỉ nhẹ 2 giây giữa các đoạn cho an toàn

                    # 4. Tạo nút tải về
                    st.success(f"✅ Dịch xong file: {f.name}")
                    st.download_button(
                        label=f"⬇️ Tải bản dịch {f.name}", 
                        data=save_docx(full_trans).getvalue(), 
                        file_name=f"VN_{f.name}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

    # --- TAB DỊCH WEB ---
    with tab2:
        urls = st.text_area("Dán danh sách Link (mỗi dòng 1 link):")
        if st.button("Bắt đầu dịch Link"):
            links = urls.split("\n")
            all_txt = ""
            progress_text = st.empty()
            
            for idx, l in enumerate(links):
                if l.strip():
                    progress_text.text(f"Đang dịch link: {l}...")
                    raw = scrape_url(l.strip())
                    
                    if len(raw) > 0:
                        try:
                            # Cắt ngắn nếu quá dài (Web thường nhiều rác)
                            res = model.generate_content(
                                f"Dịch nội dung sau sang Tiếng Việt, tóm tắt ý chính nếu quá dài:\n{raw[:20000]}", 
                                safety_settings=safety_settings
                            )
                            if res and res.text:
                                all_txt += f"\n\n--- NGUỒN: {l} ---\n{res.text}"
                        except Exception as e:
                            all_txt += f"\n--- Lỗi dịch link {l}: {e} ---\n"
                    
            st.success("Hoàn tất!")
            st.download_button("Tải file dịch tổng hợp Web", save_docx(all_txt).getvalue(), "Dich_Web.docx")

# ==============================================================================
# 3. DỊCH ẢNH (OCR)
# ==============================================================================
elif menu == "🖼️ Dịch Ảnh (OCR)":
    st.subheader("🖼️ Dịch chữ từ Hình ảnh")
    imgs = st.file_uploader("Tải ảnh lên:", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if imgs and st.button("Bắt đầu dịch ảnh"):
        full_ocr = ""
        for im_f in imgs:
            try:
                img = Image.open(im_f)
                st.image(img, caption=f"Ảnh gốc: {im_f.name}", width=300)
                
                with st.spinner(f"Đang đọc ảnh {im_f.name}..."):
                    res = model.generate_content(
                        ["Hãy trích xuất toàn bộ chữ trong ảnh này và dịch sang Tiếng Việt. Nếu là tiếng Trung/Nhật hãy dịch mượt mà:", img], 
                        safety_settings=safety_settings
                    )
                    
                    if res and res.text:
                        full_ocr += f"\n--- ẢNH: {im_f.name} ---\n" + res.text
                        st.write(res.text) # Hiện kết quả ngay
                    else:
                        st.warning(f"Không đọc được nội dung ảnh {im_f.name}")
            except Exception as e:
                st.error(f"Lỗi xử lý ảnh {im_f.name}: {e}")

        if full_ocr:
            st.download_button("Tải file dịch ảnh (.docx)", save_docx(full_ocr).getvalue(), "Dich_Anh.docx")
