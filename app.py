import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time, textwrap
from PIL import Image, ImageDraw, ImageFont, ImageOps
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Inches, Pt
from bs4 import BeautifulSoup

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & API
# ==============================================================================
st.set_page_config(page_title="Siêu AI Đa Năng Pro", page_icon="🚀", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white; font-weight: bold; width: 100%;}</style>""", unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    
    # 1. Nhập API Key (Hỗ trợ cả nhập tay và Secrets)
    api_key_input = st.text_input("🔑 API Key (Nếu chưa có trong Secrets):", type="password")
    final_api_key = api_key_input if api_key_input else st.secrets.get("GEMINI_API_KEY", "")
    
    # 2. Chọn Model (Cố định danh sách để không bị mất menu)
    model_options = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-2.0-flash-exp"]
    selected_model = st.selectbox("🧠 Chọn Model:", model_options)
    st.caption("Mẹo: 'Flash' nhanh. 'Pro' dịch hay & OCR tốt hơn.")
    
    st.divider()
    menu = st.radio("CHỨC NĂNG:", [
        "🏭 Dịch Tài Liệu Đa Năng (PDF/Word/Ảnh)", 
        "🔮 Hỏi Đáp Chuyên Sâu", 
        "🖼️ Dịch Ảnh (OCR Nhanh)",
        "🌐 Dịch Website"
    ])

# --- KẾT NỐI GEMINI ---
if not final_api_key:
    st.warning("⚠️ Vui lòng nhập Gemini API Key để bắt đầu!")
    st.stop()

try:
    genai.configure(api_key=final_api_key)
    model = genai.GenerativeModel(selected_model)
except Exception as e:
    st.error(f"❌ API Key lỗi: {e}")
    st.stop()

# Cấu hình an toàn
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ NÂNG CAO (CORE LOGIC)
# ==============================================================================

def get_font(size):
    """Tìm font hỗ trợ tiếng Việt"""
    font_paths = ["arial.ttf", "Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Calibri.ttf"]
    for path in font_paths:
        try: return ImageFont.truetype(path, size)
        except: continue
    return ImageFont.load_default()

def overlay_text_on_image(original_img, text_content):
    """Vẽ chữ đè lên ảnh (cho tranh minh họa)"""
    try:
        img = original_img.convert("RGBA")
        width, height = img.size
        # Lớp phủ mờ đen
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 160)) 
        img = Image.alpha_composite(img, overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        fontsize = max(12, int(width / 32))
        font = get_font(fontsize)
        
        # Ngắt dòng
        chars_per_line = int((width - 40) / (fontsize * 0.6))
        wrapper = textwrap.TextWrapper(width=chars_per_line)
        lines = []
        for line in text_content.split('\n'): lines.extend(wrapper.wrap(line))
        
        # Vẽ căn giữa
        text_height = len(lines) * (fontsize + 8)
        y = max(20, (height - text_height) / 2)

        for line in lines:
            bbox = font.getbbox(line) if hasattr(font, 'getbbox') else (0,0, len(line)*fontsize*0.5, fontsize)
            text_w = bbox[2] - bbox[0]
            x = max(10, (width - text_w) / 2)
            draw.text((x+2, y+2), line, font=font, fill="black")
            draw.text((x, y), line, font=font, fill=(255, 255, 100))
            y += fontsize + 8
        return img
    except: return original_img

def process_pdf_layout_preserved(file_stream):
    """Xử lý PDF giữ layout (Text + Image)"""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    content_list = [] 
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            if block["type"] == 0: # Text
                text = ""
                for line in block["lines"]:
                    for span in line["spans"]: text += span["text"] + " "
                    text += "\n"
                if text.strip(): content_list.append({"type": "text", "val": text})
            elif block["type"] == 1: # Image
                try:
                    if len(block["image"]) > 3000:
                        img_pil = Image.open(io.BytesIO(block["image"]))
                        content_list.append({"type": "image", "val": img_pil, "name": f"P{page_num}"})
                except: pass
    return content_list

def process_docx_with_images(file_stream):
    """Xử lý DOCX lấy cả Text và Ảnh"""
    doc = Document(file_stream)
    content_list = []
    for para in doc.paragraphs:
        if para.text.strip(): content_list.append({"type": "text", "val": para.text + "\n"})
        try:
            nsmap = para._element.nsmap
            blips = para._element.findall('.//a:blip', namespaces=nsmap)
            for blip in blips:
                embed_attr = blip.get(f"{{{nsmap['r']}}}embed") 
                if embed_attr:
                    image_bytes = doc.part.related_parts[embed_attr].blob
                    content_list.append({"type": "image", "val": Image.open(io.BytesIO(image_bytes)), "name": "DOCX_Img"})
        except: pass
    return content_list

def process_unified_file(uploaded_file):
    """Router xử lý mọi loại file"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf': return process_pdf_layout_preserved(uploaded_file)
    elif file_type == 'docx': return process_docx_with_images(uploaded_file)
    elif file_type in ['jpg', 'png', 'jpeg', 'webp']:
        try:
            img = Image.open(uploaded_file)
            img = ImageOps.exif_transpose(img)
            return [{"type": "image", "val": img, "name": uploaded_file.name}]
        except: pass
    elif file_type == 'txt':
        return [{"type": "text", "val": uploaded_file.getvalue().decode("utf-8")}]
    return []

def save_docx_layout(contents):
    """Lưu kết quả ra Word"""
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)
    for item in contents:
        if item['type'] == 'text':
            if item['val'].strip(): doc.add_paragraph(item['val'])
        elif item['type'] == 'image':
            img_save = item.get('val_translated', item['val']) 
            try:
                bio = io.BytesIO()
                img_save.save(bio, format='PNG')
                doc.add_picture(bio, width=Inches(5.0))
            except: pass
    bio = io.BytesIO()
    doc.save(bio)
    return bio

def scrape_url(url):
    """Hàm lấy nội dung Web cũ"""
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        tags = soup.find_all(['p', 'h1', 'h2', 'li'])
        return "\n".join([t.get_text() for t in tags])
    except: return ""

# ==============================================================================
# 3. GIAO DIỆN CHÍNH & CHỨC NĂNG
# ==============================================================================

# --- CHỨC NĂNG 1: DỊCH TÀI LIỆU ĐA NĂNG (NÂNG CẤP MẠNH NHẤT) ---
if menu == "🏭 Dịch Tài Liệu Đa Năng (PDF/Word/Ảnh)":
    st.subheader("🏭 Dịch Tài Liệu & Số Hóa (All-in-One)")
    st.info("Hỗ trợ: PDF, Word, Ảnh Scan, Ảnh Truyện. Tự động nhận diện Hán Nôm & Layout.")
    
    instr = st.text_area("Yêu cầu dịch:", value="Dịch sang tiếng Việt văn phong trang trọng, mượt mà. Giữ nguyên thuật ngữ chuyên môn.")
    up_files = st.file_uploader("Chọn file:", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'jpg', 'png', 'jpeg'])
    
    if st.button("🚀 Bắt đầu Dịch"):
        if not up_files:
            st.warning("⚠️ Chưa chọn file!")
        else:
            for f in up_files:
                st.toast(f"Đang xử lý: {f.name}")
                with st.expander(f"📄 Kết quả: {f.name}", expanded=True):
                    raw_contents = process_unified_file(f)
                    if not raw_contents:
                        st.error("Không đọc được nội dung file.")
                        continue
                    
                    final_results = []
                    text_buffer = ""
                    p_bar = st.progress(0)
                    total = len(raw_contents)
                    
                    for i, item in enumerate(raw_contents):
                        p_bar.progress((i+1)/total)
                        
                        # --- XỬ LÝ TEXT ---
                        if item['type'] == 'text':
                            text_buffer += item['val'] + "\n"
                            if len(text_buffer) < 3000 and i < total - 1 and raw_contents[i+1]['type'] == 'text': continue
                            
                            if text_buffer.strip():
                                try:
                                    res = model.generate_content(f"Dịch: {instr}\n\n{text_buffer}", safety_settings=safety_settings)
                                    final_results.append({"type": "text", "val": res.text if res else text_buffer})
                                except: final_results.append({"type": "text", "val": text_buffer})
                                text_buffer = ""

                        # --- XỬ LÝ IMAGE (HYBRID MODE) ---
                        elif item['type'] == 'image':
                            if text_buffer: # Dịch text tồn đọng
                                try:
                                    res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                                    final_results.append({"type": "text", "val": res.text})
                                except: pass
                                text_buffer = ""
                            
                            # Prompt thông minh
                            prompt = [
                                f"""Phân tích ảnh này:
                                1. Nếu là **Sách/Văn bản Scan** (Nhiều chữ): Trả về `[MODE:TEXT]` + Nội dung dịch (OCR toàn bộ).
                                2. Nếu là **Tranh minh họa** (Ít chữ): Trả về `[MODE:IMG]` + Nội dung chữ trong tranh (nếu có).
                                Yêu cầu dịch: {instr}""",
                                item['val']
                            ]
                            try:
                                res_img = model.generate_content(prompt, safety_settings=safety_settings)
                                txt_res = res_img.text if res_img else ""
                                
                                if "[MODE:TEXT]" in txt_res:
                                    final_results.append({"type": "text", "val": f"\n[Nội dung ảnh scan]:\n{txt_res.replace('[MODE:TEXT]', '')}\n"})
                                elif "[MODE:IMG]" in txt_res:
                                    caption = txt_res.replace("[MODE:IMG]", "").strip()
                                    new_img = overlay_text_on_image(item['val'], caption) if caption else item['val']
                                    final_results.append({"type": "image", "val": item['val'], "val_translated": new_img})
                                else:
                                    final_results.append({"type": "text", "val": txt_res})
                            except: final_results.append(item)

                    # Xử lý buffer cuối
                    if text_buffer:
                        try:
                            res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                            final_results.append({"type": "text", "val": res.text})
                        except: pass
                    
                    st.success("✅ Hoàn tất!")
                    st.download_button(f"⬇️ Tải Word ({f.name})", save_docx_layout(final_results).getvalue(), f"VN_{f.name}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# --- CHỨC NĂNG 2: HỎI ĐÁP (NÂNG CẤP ĐỌC FILE) ---
elif menu == "🔮 Hỏi Đáp Chuyên Sâu":
    st.subheader("🔮 Trợ Lý Chuyên Gia (Huyền Học - Data)")
    
    role = st.selectbox("Vai trò AI:", ["Đại sư Huyền học", "Chuyên gia Data", "Trợ lý đa năng"])
    
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "context" not in st.session_state: st.session_state.context = ""

    with st.expander("📚 Nạp kiến thức (PDF/Docx/Txt)"):
        edu_files = st.file_uploader("Tải tài liệu:", accept_multiple_files=True)
        if st.button("Học tài liệu") and edu_files:
            raw_text = ""
            for ef in edu_files:
                blocks = process_unified_file(ef)
                raw_text += "\n".join([b['val'] for b in blocks if b['type']=='text'])
            st.session_state.context = raw_text
            st.success(f"Đã nạp {len(raw_text)} ký tự!")

    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).markdown(m["content"])

    if q := st.chat_input("Hỏi AI..."):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        
        full_prompt = f"VAI TRÒ: {role}\nKIẾN THỨC NỀN: {st.session_state.context}\nCÂU HỎI: {q}"
        try:
            res = model.generate_content(full_prompt)
            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append({"role": "assistant", "content": res.text})
        except Exception as e: st.error(f"Lỗi: {e}")

# --- CHỨC NĂNG 3: DỊCH WEB (CŨ) ---
elif menu == "🌐 Dịch Website":
    st.subheader("🌐 Dịch Nội Dung Website")
    url = st.text_input("Nhập Link bài viết:")
    if st.button("Dịch ngay") and url:
        with st.spinner("Đang cào và dịch..."):
            raw = scrape_url(url)
            if raw:
                try:
                    res = model.generate_content(f"Dịch bài này sang tiếng Việt:\n{raw[:15000]}", safety_settings=safety_settings)
                    st.markdown(res.text)
                    st.download_button("Tải kết quả", res.text, "Web_Trans.txt")
                except Exception as e: st.error(f"Lỗi dịch: {e}")
            else: st.error("Không lấy được nội dung web này.")

# --- CHỨC NĂNG 4: DỊCH ẢNH LẺ (CŨ) ---
elif menu == "🖼️ Dịch Ảnh (OCR Nhanh)":
    st.subheader("🖼️ Công cụ Dịch Ảnh Nhanh")
    imgs = st.file_uploader("Tải ảnh:", accept_multiple_files=True, type=['jpg', 'png'])
    if imgs:
        for f in imgs:
            img = Image.open(f)
            c1, c2 = st.columns(2)
            c1.image(img, caption="Gốc")
            if st.button(f"Dịch {f.name}"):
                res = model.generate_content(["Dịch nội dung trong ảnh sang Tiếng Việt:", img], safety_settings=safety_settings)
                if res:
                    c2.image(overlay_text_on_image(img, res.text), caption="Đã dịch")
                    st.write(res.text)
