import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Inches
from bs4 import BeautifulSoup

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Siêu AI Đa Năng", page_icon="🚀", layout="wide")

# --- CẤU HÌNH AN TOÀN ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- KẾT NỐI API ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ Chưa cấu hình GEMINI_API_KEY trong Secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Lấy danh sách model có sẵn
try:
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    default_model = "models/gemini-1.5-flash" if "models/gemini-1.5-flash" in available_models else available_models[0]
except:
    default_model = "models/gemini-1.5-flash"

# --- HÀM XỬ LÝ ---

def process_pdf_mixed(file_bytes):
    """Tách chữ và ảnh từ PDF"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    content_list = []

    for page_num, page in enumerate(doc):
        # 1. Lấy chữ
        text = page.get_text().strip()
        if text:
            content_list.append({"type": "text", "val": text})

        # 2. Lấy ảnh
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if len(image_bytes) > 10240: # Chỉ lấy ảnh > 10KB
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    content_list.append({"type": "image", "val": img_pil})
            except:
                continue
    return content_list

def save_docx_mixed(contents):
    """Lưu kết quả ra file Word"""
    doc = Document()
    for item in contents:
        if item['type'] == 'text':
            doc.add_paragraph(item['val'])
        elif item['type'] == 'image':
            img_io = io.BytesIO()
            item['val'].save(img_io, format='PNG')
            doc.add_picture(img_io, width=Inches(5))
            if 'trans' in item:
                p = doc.add_paragraph()
                p.add_run(f"\n[DỊCH ẢNH]: {item['trans']}").italic = True
    
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# --- GIAO DIỆN ---
st.title("🚀 Siêu Trợ Lý Dịch Thuật Đa Năng")

with st.sidebar:
    selected_model = st.selectbox("Chọn Model:", [default_model] + ["models/gemini-1.5-pro"])
    menu = st.radio("CHỨC NĂNG:", ["🏭 Dịch Thuật Công Nghiệp", "🔮 Hỏi Đáp", "🖼️ Dịch Ảnh (OCR)"])

model = genai.GenerativeModel(selected_model)

if menu == "🏭 Dịch Thuật Công Nghiệp":
    st.subheader("🏭 Dịch Sách/Tài liệu (Hỗ trợ PDF Mixed)")
    
    instr = st.text_area("Yêu cầu dịch:", "Dịch sang tiếng Việt mượt mà, văn phong chuyên nghiệp.")
    gloss = st.text_area("Thuật ngữ (Glossary):", "VD: Trúc Cơ -> Foundation Establishment")
    
    up_files = st.file_uploader("Tải tệp lên (PDF, Docx, TXT):", accept_multiple_files=True)
    
    if st.button("🚀 Bắt đầu dịch") and up_files:
        for f in up_files:
            st.info(f" đang xử lý: {f.name}")
            file_bytes = f.read() # Đọc bytes một lần duy nhất
            
            # Phân tích nội dung
            raw_contents = []
            if f.name.endswith('.pdf'):
                raw_contents = process_pdf_mixed(file_bytes)
            else:
                # Xử lý đơn giản cho TXT/Docx (Chỉ lấy chữ)
                raw_contents = [{"type": "text", "val": file_bytes.decode('utf-8', errors='ignore')}]

            final_results = []
            p_bar = st.progress(0)
            
            for idx, item in enumerate(raw_contents):
                try:
                    if item['type'] == 'text':
                        prompt = f"{instr}\nThuật ngữ: {gloss}\nNội dung:\n{item['val']}"
                        res = model.generate_content(prompt, safety_settings=safety_settings)
                        final_results.append({"type": "text", "val": res.text if res else "[Lỗi dịch đoạn này]"})
                    
                    elif item['type'] == 'image':
                        prompt = [f"Dịch chữ trong ảnh này sang tiếng Việt. {instr}", item['val']]
                        res = model.generate_content(prompt, safety_settings=safety_settings)
                        final_results.append({"type": "image", "val": item['val'], "trans": res.text if res else ""})
                    
                    p_bar.progress((idx + 1) / len(raw_contents))
                    time.sleep(1) # Tránh Rate Limit
                except Exception as e:
                    st.error(f"Lỗi tại phần tử {idx}: {e}")
                    time.sleep(5)

            # Xuất file
            docx_data = save_docx_mixed(final_results)
            st.download_button(f"⬇️ Tải bản dịch {f.name}", docx_data, f"Dich_{f.name}.docx")

elif menu == "🖼️ Dịch Ảnh (OCR)":
    imgs = st.file_uploader("Tải ảnh:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if imgs and st.button("Dịch ngay"):
        for im_f in imgs:
            img = Image.open(im_f)
            st.image(img, width=400)
            with st.spinner("AI đang đọc ảnh..."):
                res = model.generate_content(["Dịch toàn bộ chữ trong ảnh sang tiếng Việt:", img], safety_settings=safety_settings)
                st.write(res.text)

# (Phần Hỏi Đáp bạn có thể giữ nguyên logic cũ nhưng hãy dùng file_bytes.read() cẩn thận)
