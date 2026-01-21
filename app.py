import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time
from PIL import Image
# Thay PyPDF2 bằng PyMuPDF (fitz) để xử lý ảnh tốt hơn
import fitz  
from docx import Document
from docx.shared import Inches
from bs4 import BeautifulSoup

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Siêu AI Hán Nôm & Dịch Thuật", page_icon="☯️", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #8e44ad; color: white;}</style>""", unsafe_allow_html=True)

# --- CẤU HÌNH AN TOÀN ---
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
    # Ưu tiên Flash 1.5 cho tốc độ và Pro 1.5 cho độ chính xác Hán Nôm
    available_models = ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]
except:
    st.error("⚠️ Chưa nhập API Key trong Secrets.")
    st.stop()

# --- CÁC HÀM XỬ LÝ CỐT LÕI ---

def extract_content_from_pdf(uploaded_file):
    """
    Hàm này đọc PDF và tách riêng:
    1. Văn bản (Text)
    2. Hình ảnh (Images)
    Trả về một danh sách các 'Block' để giữ đúng thứ tự trang.
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    content_blocks = [] # Chứa {type: 'text'/'image', content: ...}

    for page_num, page in enumerate(doc):
        # 1. Lấy văn bản của trang
        text = page.get_text()
        if text.strip():
            content_blocks.append({
                "type": "text", 
                "page": page_num + 1, 
                "content": text
            })

        # 2. Lấy hình ảnh của trang
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Lọc bỏ ảnh quá nhỏ (logo, icon, đường kẻ) - Dưới 5KB bỏ qua
            if len(image_bytes) > 5120: 
                image_pil = Image.open(io.BytesIO(image_bytes))
                content_blocks.append({
                    "type": "image",
                    "page": page_num + 1,
                    "content": image_pil,
                    "name": f"Trang_{page_num+1}_Anh_{img_index+1}"
                })
    return content_blocks

def save_docx_mixed(blocks, translation_results):
    """
    Tạo file Word chứa cả Ảnh và Văn bản đã dịch
    """
    doc = Document()
    doc.add_heading('BẢN DỊCH TÀI LIỆU', 0)

    for i, block in enumerate(blocks):
        # Nếu là Text
        if block['type'] == 'text':
            # Tìm bản dịch tương ứng trong results (dựa vào index)
            if i < len(translation_results) and translation_results[i]:
                doc.add_paragraph(translation_results[i])
                doc.add_paragraph("-" * 20) # Đường kẻ phân cách
        
        # Nếu là Image
        elif block['type'] == 'image':
            img_pil = block['content']
            
            # 1. Chèn ảnh gốc vào Word
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format=img_pil.format if img_pil.format else 'PNG')
            doc.add_picture(img_byte_arr, width=Inches(4.0)) # Chèn ảnh rộng 4 inch
            
            # 2. Chèn bản dịch nội dung trong ảnh ngay bên dưới
            if i < len(translation_results) and translation_results[i]:
                p = doc.add_paragraph()
                runner = p.add_run(f"\n[DỊCH ẢNH TRÊN]:\n{translation_results[i]}")
                runner.bold = True
                runner.italic = True
                doc.add_paragraph("-" * 20)

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
st.title("☯️ Siêu AI: Dịch Hán Nôm & Tài Liệu Cổ")

with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.info("💡 Mẹo: Chọn 'Gemini 1.5 Pro' để dịch Hán Nôm dọc tốt nhất.")
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["🏭 Dịch Tài Liệu (PDF/Hán Nôm/Ảnh)", "🔮 Hỏi Đáp Hán Học", "🖼️ Dịch Ảnh Rời (OCR)"])

model = genai.GenerativeModel(selected_model)

# ==============================================================================
# 1. DỊCH TÀI LIỆU (PDF CHỨA ẢNH & CHỮ)
# ==============================================================================
if menu == "🏭 Dịch Tài Liệu (PDF/Hán Nôm/Ảnh)":
    st.subheader("📜 Dịch PDF chứa Ảnh Minh Họa / Sách Cổ")
    st.markdown("""
    **Tính năng đặc biệt:**
    - Tự động tách ảnh từ PDF.
    - Nếu là ảnh sách cổ (chữ Hán dọc, phải sang trái) -> AI tự xoay chiều dịch sang tiếng Việt ngang.
    - Kết quả xuất ra file Word: **[Hình Ảnh]** kèm **[Bản Dịch]** ngay bên dưới.
    """)
    
    instr = st.text_area("Yêu cầu dịch:", value="Dịch sang tiếng Việt hiện đại, văn phong trang trọng. Nếu là thơ giữ nguyên thể thơ.")
    
    uploaded_file = st.file_uploader("Tải file PDF:", type=['pdf'])
    
    if uploaded_file and st.button("🚀 Bắt đầu Phân Tích & Dịch"):
        st.info("⏳ Đang tách bóc nội dung (Chữ và Ảnh) từ PDF...")
        
        # 1. Tách nội dung
        try:
            blocks = extract_content_from_pdf(uploaded_file)
            st.success(f"✅ Đã tìm thấy: {len([b for b in blocks if b['type']=='text'])} đoạn văn bản và {len([b for b in blocks if b['type']=='image'])} hình ảnh.")
        except Exception as e:
            st.error(f"Lỗi đọc PDF: {e}")
            st.stop()

        # 2. Xử lý dịch từng block
        translation_results = []
        p_bar = st.progress(0)
        
        for i, block in enumerate(blocks):
            res_text = ""
            
            # --- TRƯỜNG HỢP 1: LÀ VĂN BẢN (TEXT) ---
            if block['type'] == 'text':
                content = block['content']
                # Gộp prompt
                prompt = f"YÊU CẦU: {instr}\nNỘI DUNG CẦN DỊCH:\n{content[:5000]}" # Cắt 5000 ký tự an toàn
                
                # Logic thử lại 3 lần
                for attempt in range(3):
                    try:
                        res = model.generate_content(prompt, safety_settings=safety_settings)
                        if res and res.text:
                            res_text = res.text
                            break
                    except Exception as e:
                        if "ResourceExhausted" in str(e): time.sleep(20)
                        else: time.sleep(2)
            
            # --- TRƯỜNG HỢP 2: LÀ HÌNH ẢNH (IMAGE) ---
            elif block['type'] == 'image':
                img = block['content']
                # Prompt đặc biệt cho Hán Nôm / Sách cổ
                prompt_img = [
                    f"""
                    Hãy phân tích hình ảnh này. 
                    1. Nếu đây là trang sách chữ Hán (viết dọc, từ phải sang trái): Hãy nhận diện chữ, phiên âm Hán Việt và dịch nghĩa sang tiếng Việt hiện đại (viết ngang, trái sang phải).
                    2. Nếu đây là hình minh họa có chữ: Hãy dịch tất cả chữ trong hình.
                    3. YÊU CẦU BỔ SUNG: {instr}
                    """,
                    img
                ]
                
                for attempt in range(3):
                    try:
                        res = model.generate_content(prompt_img, safety_settings=safety_settings)
                        if res and res.text:
                            res_text = res.text
                            break
                    except Exception as e:
                        if "ResourceExhausted" in str(e): time.sleep(20)
                        else: time.sleep(2)

            # Lưu kết quả
            if res_text:
                translation_results.append(res_text)
                st.toast(f"✅ Xong phần {i+1}/{len(blocks)}")
            else:
                translation_results.append("[Không dịch được phần này]")
            
            p_bar.progress((i+1)/len(blocks))
            time.sleep(1) # Nghỉ nhẹ

        # 3. Xuất file
        st.success("🎉 Hoàn tất dịch thuật!")
        docx_file = save_docx_mixed(blocks, translation_results)
        
        st.download_button(
            label="⬇️ Tải bản dịch Word (.docx)",
            data=docx_file.getvalue(),
            file_name=f"Dich_Han_Nom_{uploaded_file.name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# ==============================================================================
# 2. HỎI ĐÁP HÁN HỌC
# ==============================================================================
elif menu == "🔮 Hỏi Đáp Hán Học":
    st.subheader("🔮 Giải Nghĩa Hán Nôm & Phong Thủy")
    
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).markdown(m["content"])

    if q := st.chat_input("Nhập câu đối, đoạn văn Hán cổ cần giải nghĩa..."):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        
        prompt = f"Bạn là một chuyên gia Hán Nôm và Huyền học. Hãy giải thích chi tiết đoạn sau (Phiên âm, Dịch nghĩa, Điển tích nếu có):\n{q}"
        
        with st.spinner("Đang luận giải..."):
            try:
                res = model.generate_content(prompt, safety_settings=safety_settings)
                st.chat_message("assistant").markdown(res.text)
                st.session_state.chat_history.append({"role": "assistant", "content": res.text})
            except Exception as e: st.error(f"Lỗi: {e}")

# ==============================================================================
# 3. DỊCH ẢNH RỜI (OCR)
# ==============================================================================
elif menu == "🖼️ Dịch Ảnh Rời (OCR)":
    st.subheader("🖼️ Upload Ảnh Lẻ (JPG/PNG)")
    imgs = st.file_uploader("Tải ảnh lên:", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if imgs and st.button("Dịch Ngay"):
        for im_f in imgs:
            img = Image.open(im_f)
            st.image(img, width=300)
            with st.spinner(f"Đang dịch {im_f.name}..."):
                try:
                    res = model.generate_content(
                        ["Nhận diện chữ Hán/Nôm (kể cả viết dọc) và dịch sang Tiếng Việt:", img], 
                        safety_settings=safety_settings
                    )
                    st.write(res.text)
                except Exception as e:
                    st.error(f"Lỗi: {e}")
