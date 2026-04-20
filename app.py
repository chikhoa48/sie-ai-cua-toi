import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io, time, random
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Inches

# --- CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="Siêu Trợ Lý Đa Năng 2026", layout="wide", page_icon="🚀")

# --- HÀM LẤY MODEL MỚI NHẤT ---
def get_models(api_key):
    try:
        genai.configure(api_key=api_key)
        return [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except:
        return ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cài đặt hệ thống")
    keys = st.secrets.get("GEMINI_KEYS", [st.secrets.get("GEMINI_API_KEY")])
    
    # Chọn model
    available_models = get_models(keys[0])
    selected_model = st.selectbox("Chọn Model Gemini:", available_models)
    custom_model = st.text_input("Hoặc nhập mã model (VD: models/gemini-3.1-pro):")
    final_model = custom_model if custom_model else selected_model
    
    st.divider()
    instr = st.text_area("Yêu cầu dịch:", "Dịch sang tiếng Việt mượt mà, văn phong chuyên nghiệp.")
    glossary = st.text_area("Từ điển thuật ngữ:", "AI -> Trí tuệ nhân tạo")

# --- ENGINE DỊCH THUẬT CHỐNG NGHẼN ---
def translate_core(prompt_parts):
    # Xoay vòng key ngẫu nhiên để tăng hạn mức
    genai.configure(api_key=random.choice(keys))
    model = genai.GenerativeModel(final_model)
    
    for attempt in range(10): # Thử lại tối đa 10 lần
        try:
            res = model.generate_content(prompt_parts, safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            })
            return res.text
        except Exception as e:
            if "429" in str(e):
                wait = 30 + (attempt * 10)
                st.toast(f"⏳ Đang nghẽn mạch, nghỉ {wait}s...", icon="💤")
                time.sleep(wait)
            else:
                st.error(f"Lỗi: {e}")
                time.sleep(5)
    return "[Lỗi: Hết hạn mức hoàn toàn]"

# --- HÀM XỬ LÝ VĂN BẢN 30.000 TỪ ---
def process_text_large(text):
    # Chia nhỏ 30k từ thành các block 10k ký tự
    chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
    results = []
    for c in chunks:
        res = translate_core(f"{instr}\nThuật ngữ: {glossary}\n\nNội dung:\n{c}")
        results.append(res)
    return "\n".join(results)

# --- TRÍCH XUẤT ĐA ĐỊNH DẠNG ---
def extract_content(uploaded_file):
    name = uploaded_file.name.lower()
    content = []

    # 1. Xử lý PDF
    if name.endswith('.pdf'):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text = page.get_text().strip()
            if text: content.append({"type": "text", "val": text})
            for img in page.get_images():
                try:
                    base = doc.extract_image(img[0])
                    if base["size"] > 15000:
                        content.append({"type": "image", "val": Image.open(io.BytesIO(base["image"]))})
                except: pass

    # 2. Xử lý DOCX
    elif name.endswith('.docx'):
        doc = Document(uploaded_file)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        content.append({"type": "text", "val": full_text})

    # 3. Xử lý TXT
    elif name.endswith('.txt'):
        text = uploaded_file.getvalue().decode("utf-8")
        content.append({"type": "text", "val": text})

    # 4. Xử lý Hình ảnh trực tiếp
    elif name.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(uploaded_file)
        content.append({"type": "image", "val": img})
        
    return content

# --- CHƯƠNG TRÌNH CHÍNH ---
st.title("🛡️ Siêu Trợ Lý Dịch Thuật Đa Định Dạng")
st.write("Hỗ trợ: **PDF, Word, Text, Hình ảnh (OCR)**")

up_files = st.file_uploader("Nạp tệp tin của bạn tại đây:", accept_multiple_files=True)

if st.button("🚀 BẮT ĐẦU DỊCH TẤT CẢ") and up_files:
    final_output = []
    
    for f in up_files:
        st.info(f"📂 Đang xử lý: {f.name}")
        items = extract_content(f)
        
        p_bar = st.progress(0)
        for i, item in enumerate(items):
            if item['type'] == 'text':
                translated = process_text_large(item['val'])
                final_output.append({"type": "text", "val": translated})
            elif item['type'] == 'image':
                trans_img = translate_core([f"Dịch các chữ trong ảnh này: {instr}", item['val']])
                final_output.append({"type": "image", "val": item['val'], "trans": trans_img})
            p_bar.progress((i + 1) / len(items))

    # Xuất file Word duy nhất
    out_doc = Document()
    out_doc.add_heading('BẢN DỊCH TỔNG HỢP', 0)
    for res in final_output:
        if res['type'] == 'text':
            out_doc.add_paragraph(res['val'])
        else:
            img_io = io.BytesIO()
            res['val'].save(img_io, format='PNG')
            out_doc.add_picture(img_io, width=Inches(5))
            out_doc.add_paragraph(f"[Bản dịch ảnh]: {res['trans']}").italic = True
            
    bio = io.BytesIO()
    out_doc.save(bio)
    st.download_button("⬇️ Tải xuống kết quả (.docx)", bio.getvalue(), "Ket_Qua_Dich.docx")
    st.success("🎉 Đã hoàn thành toàn bộ!")
