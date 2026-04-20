import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io, time
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Inches

# --- CẤU HÌNH HỆ THỐNG 2026 ---
st.set_page_config(page_title="Gemini 3.1 Ultra Translator", layout="wide")

# Thiết lập Model - Cập nhật tên model theo phiên bản mới nhất bạn có
# Ví dụ: "models/gemini-3.1-pro", "models/gemini-3.1-flash"
MODEL_ID = "models/gemini-3.1-pro" 

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except:
    st.error("⚠️ Vui lòng cấu hình GEMINI_API_KEY!")
    st.stop()

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- ENGINE DỊCH THUẬT 30.000 TỪ ---

def process_large_text(text, instruction, glossary):
    """
    Chia nhỏ 30.000 từ thành các block lớn để dịch.
    Mỗi block khoảng 15.000 - 20.000 ký tự (~4000 từ) 
    để đảm bảo kết quả trả về (output) không bị cắt ngang.
    """
    # Gemini 3.1 có context window cực lớn, nhưng output limit vẫn là điểm cần lưu ý
    max_chunk_size = 20000 
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    translated_chunks = []
    model = genai.GenerativeModel(MODEL_ID)
    
    progress_text = st.empty()
    
    for idx, chunk in enumerate(chunks):
        progress_text.text(f"⏳ Đang dịch khối {idx+1}/{len(chunks)}...")
        
        prompt = f"""
        VAI TRÒ: Chuyên gia dịch thuật ngôn ngữ cao cấp.
        NHIỆM VỤ: Dịch văn bản sau sang Tiếng Việt.
        YÊU CẦU: {instruction}
        THUẬT NGỮ CẦN TUÂN THỦ: {glossary}
        
        NỘI DUNG GỐC:
        {chunk}
        """
        
        success = False
        retries = 0
        while not success and retries < 5:
            try:
                response = model.generate_content(prompt, safety_settings=safety_settings)
                if response.text:
                    translated_chunks.append(response.text)
                    success = True
            except Exception as e:
                retries += 1
                wait_time = 15 * retries
                st.warning(f"⚠️ Lỗi Quota/Kết nối. Đang đợi {wait_time}s (Lần thử {retries})...")
                time.sleep(wait_time)
        
        # Nghỉ ngắn giữa các block để tối ưu hóa API
        time.sleep(2)
        
    return "\n".join(translated_chunks)

# --- XỬ LÝ FILE ĐA PHƯƠNG TIỆN ---

def extract_pdf_content(f):
    file_bytes = f.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    elements = []
    
    for page in doc:
        # Lấy văn bản
        text = page.get_text().strip()
        if text:
            elements.append({"type": "text", "val": text})
        
        # Lấy hình ảnh (Hỗ trợ Hán Nôm/Hình ảnh minh họa)
        for img in page.get_images():
            try:
                base_img = doc.extract_image(img[0])
                pil_img = Image.open(io.BytesIO(base_img["image"]))
                if base_img["size"] > 20000: # Chỉ lấy ảnh chất lượng cao
                    elements.append({"type": "image", "val": pil_img})
            except: pass
    return elements

# --- GIAO DIỆN ---
st.title("🛡️ Gemini 3.1 Pro: Industrial Translation System")
st.caption(f"Phiên bản tối ưu cho tài liệu lớn | Model: {MODEL_ID}")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Cấu hình")
    instruction = st.text_area("Chỉ thị dịch nâng cao:", "Dịch thuật học thuật, văn phong trang trọng, mượt mà.")
    glossary = st.text_area("Từ điển (Glossary):", "AI -> Trí tuệ nhân tạo\nBlockchain -> Chuỗi khối")
    
up_files = st.file_uploader("Nạp tài liệu (PDF lên đến 30.000 từ):", accept_multiple_files=True)

if st.button("🚀 BẮT ĐẦU XỬ LÝ") and up_files:
    for f in up_files:
        with st.status(f"Đang xử lý {f.name}...", expanded=True) as status:
            elements = extract_pdf_content(f)
            st.write(f"🔍 Tìm thấy {len(elements)} phân đoạn (văn bản & hình ảnh).")
            
            final_content = []
            
            for item in elements:
                if item['type'] == 'text':
                    # Sử dụng engine dịch block lớn
                    translated_val = process_large_text(item['val'], instruction, glossary)
                    final_content.append({"type": "text", "val": translated_val})
                
                elif item['type'] == 'image':
                    st.write("🖼️ Đang dịch văn bản trong ảnh...")
                    model = genai.GenerativeModel(MODEL_ID)
                    res = model.generate_content([f"Dịch chữ trong ảnh này: {instruction}", item['val']], safety_settings=safety_settings)
                    final_content.append({"type": "image", "val": item['val'], "trans": res.text})
            
            status.update(label="✅ Đã dịch xong!", state="complete")

            # Tạo file Word kết quả
            doc_out = Document()
            for c in final_content:
                if c['type'] == 'text':
                    doc_out.add_paragraph(c['val'])
                else:
                    img_io = io.BytesIO()
                    c['val'].save(img_io, format='PNG')
                    doc_out.add_picture(img_io, width=Inches(5))
                    doc_out.add_paragraph(f"[Bản dịch ảnh]: {c['trans']}")
            
            bio = io.BytesIO()
            doc_out.save(bio)
            st.download_button(f"⬇️ Tải file dịch {f.name}", bio.getvalue(), f"Dich_3.1_{f.name}.docx")
