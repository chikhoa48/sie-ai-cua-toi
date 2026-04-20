import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io, time, random
from PIL import Image
import fitz
from docx import Document
from docx.shared import Inches

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Gemini Custom Translator", layout="wide", page_icon="⚙️")

# --- QUẢN LÝ API KEY ---
def get_api_keys():
    # Lấy danh sách key từ Secrets (hỗ trợ cả 1 key hoặc nhiều key)
    keys = st.secrets.get("GEMINI_KEYS", [])
    if not keys:
        key_single = st.secrets.get("GEMINI_API_KEY")
        if key_single: keys = [key_single]
    return keys

# --- LẤY DANH SÁCH MODEL KHẢ DỤNG ---
def fetch_available_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return sorted(models, reverse=True)
    except:
        return ["models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp"]

# --- GIAO DIỆN THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ CẤU HÌNH MODEL")
    
    keys = get_api_keys()
    if not keys:
        st.error("❌ Chưa nhập API Key vào Secrets!")
        st.stop()
    
    # Nút chọn Model
    base_models = fetch_available_models(keys[0])
    selected_model = st.selectbox("Chọn phiên bản Gemini:", base_models, index=0)
    
    # Cho phép nhập thủ công nếu là bản 3.1 hoặc bản mới hơn
    custom_model = st.text_input("Hoặc nhập tên Model thủ công:", placeholder="models/gemini-3.1-pro")
    
    final_model_name = custom_model if custom_model else selected_model
    st.info(f"Đang dùng: **{final_model_name}**")
    
    st.divider()
    st.header("📝 CHỈ THỊ DỊCH")
    instr = st.text_area("Yêu cầu:", "Dịch sang tiếng Việt mượt mà, giữ nguyên thuật ngữ chuyên môn.")
    glossary = st.text_area("Từ điển thuật ngữ:", "AI -> Trí tuệ nhân tạo")

# --- ENGINE DỊCH THUẬT ---
def translate_engine(prompt_data):
    # Xoay vòng Key để chống lỗi hạn mức (Quota)
    current_key = random.choice(keys)
    genai.configure(api_key=current_key)
    model = genai.GenerativeModel(final_model_name)
    
    safety = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    for attempt in range(5):
        try:
            res = model.generate_content(prompt_data, safety_settings=safety)
            if res and res.text:
                return res.text
        except Exception as e:
            if "429" in str(e):
                wait = 20 + (attempt * 10)
                st.toast(f"⏳ Hết hạn mức, đợi {wait}s...", icon="⏳")
                time.sleep(wait)
            else:
                st.error(f"Lỗi: {e}")
                time.sleep(5)
    return "[Lỗi dịch đoạn này]"

# --- XỬ LÝ VĂN BẢN LỚN (30.000 TỪ) ---
def process_text_30k(text):
    # Chia nhỏ 30.000 từ thành các đoạn 10.000 ký tự (~3000 từ mỗi đoạn)
    chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
    translated_parts = []
    for c in chunks:
        res = translate_engine(f"{instr}\nThuật ngữ: {glossary}\n\nNội dung:\n{c}")
        translated_parts.append(res)
    return "\n".join(translated_parts)

# --- CHƯƠNG TRÌNH CHÍNH ---
st.title("🛡️ Siêu Trợ Lý Dịch Thuật Đa Phiên Bản")

uploaded_file = st.file_uploader("Tải lên PDF (Hỗ trợ tài liệu cực lớn):", type="pdf")

if uploaded_file and st.button("🚀 BẮT ĐẦU DỊCH"):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    final_results = []
    
    progress = st.progress(0)
    for i, page in enumerate(doc):
        # 1. Dịch chữ
        text = page.get_text().strip()
        if text:
            translated_text = process_text_30k(text)
            final_results.append({"type": "text", "val": translated_text})
        
        # 2. Dịch ảnh
        for img_info in page.get_images():
            try:
                base = doc.extract_image(img_info[0])
                pil_img = Image.open(io.BytesIO(base["image"]))
                if base["size"] > 15000: # Bỏ qua ảnh rác
                    trans_img = translate_engine([f"Dịch chữ trong ảnh này: {instr}", pil_img])
                    final_results.append({"type": "image", "val": pil_img, "trans": trans_img})
            except: pass
            
        progress.progress((i + 1) / len(doc))

    # Tạo file Word
    out_doc = Document()
    for item in final_results:
        if item['type'] == 'text':
            out_doc.add_paragraph(item['val'])
        else:
            img_io = io.BytesIO()
            item['val'].save(img_io, format='PNG')
            out_doc.add_picture(img_io, width=Inches(5))
            out_doc.add_paragraph(f"[Bản dịch ảnh]: {item['trans']}").italic = True
            
    bio = io.BytesIO()
    out_doc.save(bio)
    st.download_button("⬇️ Tải xuống bản dịch (.docx)", bio.getvalue(), f"Dich_{final_model_name.replace('/', '_')}.docx")
