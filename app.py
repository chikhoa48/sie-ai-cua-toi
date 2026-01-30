import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time, textwrap
from PIL import Image, ImageDraw, ImageFont, ImageOps
# Thư viện xử lý PDF & Word
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Inches, Pt
from bs4 import BeautifulSoup

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Siêu AI Đa Năng", page_icon="🚀", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white; font-weight: bold;}</style>""", unsafe_allow_html=True)

# --- CẤU HÌNH AN TOÀN (GIỮ NGUYÊN) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- KẾT NỐI API & LẤY MODEL (KHÔI PHỤC LOGIC CŨ) ---
# Thêm ô nhập Key dự phòng nếu file secrets chưa cấu hình
if "GEMINI_API_KEY" not in st.secrets:
    st.sidebar.warning("⚠️ Chưa có file secrets. Nhập Key bên dưới:")
    api_key_input = st.sidebar.text_input("Gemini API Key:", type="password")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        genai.configure(api_key=api_key_input)
else:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

# Logic lấy danh sách model động (Như code cũ của bạn)
available_models = []
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
            available_models.append(m.name)
except Exception as e:
    st.sidebar.error(f"Không lấy được danh sách Model: {e}")

# Fallback nếu không lấy được list
if not available_models: 
    available_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
else:
    available_models.sort(key=lambda x: "flash" not in x) # Ưu tiên Flash lên đầu

# ==============================================================================
# 1. CÁC HÀM HỖ TRỢ MỚI (XỬ LÝ ẢNH, FONT, FILE ĐA NĂNG)
# ==============================================================================

def get_font(size):
    """Tìm font tiếng Việt"""
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
        
        text_height = len(lines) * (fontsize + 8)
        y = max(20, (height - text_height) / 2)

        for line in lines:
            try: bbox = font.getbbox(line)
            except: bbox = (0,0, len(line)*10, 20)
            text_w = bbox[2] - bbox[0]
            x = max(10, (width - text_w) / 2)
            
            draw.text((x+2, y+2), line, font=font, fill="black")
            draw.text((x, y), line, font=font, fill=(255, 255, 100))
            y += fontsize + 8
        return img
    except: return original_img

def process_unified_file(uploaded_file):
    """
    HÀM XỬ LÝ ĐA NĂNG MỚI (Thay thế hàm process_pdf_mixed cũ)
    Hỗ trợ: PDF, Word, Ảnh, Text
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    content_list = []

    # 1. PDF (Dùng logic giữ Layout)
    if file_type == 'pdf':
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block["type"] == 0: # Text
                    txt = ""
                    for line in block["lines"]:
                        for span in line["spans"]: txt += span["text"] + " "
                        txt += "\n"
                    if txt.strip(): content_list.append({"type": "text", "val": txt})
                elif block["type"] == 1: # Image
                    try:
                        if len(block["image"]) > 3000:
                            img = Image.open(io.BytesIO(block["image"]))
                            content_list.append({"type": "image", "val": img, "name": f"PDF_P{page_num}"})
                    except: pass
    
    # 2. WORD (DOCX - Lấy cả ảnh)
    elif file_type == 'docx':
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            if para.text.strip(): content_list.append({"type": "text", "val": para.text + "\n"})
            try:
                nsmap = para._element.nsmap
                blips = para._element.findall('.//a:blip', namespaces=nsmap)
                for blip in blips:
                    embed_attr = blip.get(f"{{{nsmap['r']}}}embed") 
                    if embed_attr:
                        img_bytes = doc.part.related_parts[embed_attr].blob
                        content_list.append({"type": "image", "val": Image.open(io.BytesIO(img_bytes)), "name": "DOCX_Img"})
            except: pass

    # 3. ẢNH (JPG/PNG)
    elif file_type in ['jpg', 'png', 'jpeg', 'webp']:
        try:
            img = Image.open(uploaded_file)
            img = ImageOps.exif_transpose(img)
            content_list.append({"type": "image", "val": img, "name": uploaded_file.name})
        except: pass

    # 4. TEXT
    elif file_type == 'txt':
        content_list.append({"type": "text", "val": uploaded_file.getvalue().decode("utf-8")})

    return content_list

def save_docx_mixed(contents):
    """Hàm lưu Word nâng cấp (Hỗ trợ ảnh đã vẽ đè)"""
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)
    
    for item in contents:
        if item['type'] == 'text':
            for line in item['val'].split('\n'):
                if line.strip(): doc.add_paragraph(line)
        elif item['type'] == 'image':
            # Ưu tiên lấy ảnh đã vẽ đè (val_translated), nếu không thì lấy ảnh gốc
            img_to_save = item.get('val_translated', item['val'])
            try:
                img_byte = io.BytesIO()
                img_to_save.save(img_byte, format='PNG')
                doc.add_picture(img_byte, width=Inches(5.0))
                
                # Nếu là ảnh scan đã chuyển thành text -> ghi chú thích
                if 'trans_text' in item and "[MODE:TEXT]" in item.get('mode_tag', ''):
                     p = doc.add_paragraph("--- [Nội dung trích xuất từ ảnh] ---")
                     p.italic = True
            except: pass
            
    bio = io.BytesIO()
    doc.save(bio)
    return bio

def scrape_url(url):
    """Hàm cào web cũ"""
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
    # Menu chọn model cũ
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.caption("Mẹo: Dùng 'Flash' để dịch nhanh, 'Pro' để thông minh hơn.")
    st.divider()
    # Menu chức năng cũ
    menu = st.radio("CHỨC NĂNG:", ["🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)", "🏭 Dịch Thuật Công Nghiệp", "🖼️ Dịch Ảnh (OCR)"])

model = genai.GenerativeModel(selected_model)

# ==============================================================================
# 1. HỎI ĐÁP CHUYÊN SÂU (GIỮ NGUYÊN + NÂNG CẤP ĐỌC FILE)
# ==============================================================================
if menu == "🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)":
    st.subheader("🔮 Trợ Lý Chuyên Gia")
    
    with st.sidebar:
        role = st.selectbox("Vai trò AI:", ["Đại sư Huyền học (Giang Công)", "Chuyên gia Marketing & Data", "Trợ lý đa năng"])
        # Nâng cấp: Chấp nhận nhiều loại file hơn cho RAG
        files = st.file_uploader("Nạp tài liệu (PDF/Docx/Txt):", accept_multiple_files=True)
        if st.button("Nạp vào bộ não"):
            if files:
                raw_txt = ""
                for f in files:
                    # Dùng hàm mới để đọc text từ mọi loại file
                    blocks = process_unified_file(f)
                    for b in blocks:
                        if b['type'] == 'text': raw_txt += b['val'] + "\n"
                st.session_state.context = raw_txt
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
            except Exception as e: st.error(f"Lỗi: {e}")

# ==============================================================================
# 2. DỊCH THUẬT CÔNG NGHIỆP (NÂNG CẤP LÕI XỬ LÝ)
# ==============================================================================
elif menu == "🏭 Dịch Thuật Công Nghiệp":
    st.subheader("🏭 Dịch Sách & Truyện Hàng Loạt (Hỗ trợ Ảnh & Hán Nôm)")
    
    st.markdown("ℹ️ **Cơ chế Mới:** Hỗ trợ PDF, Word, Ảnh Scan. Tự động nhận diện sách Hán Nôm (đọc dọc) hoặc tranh minh họa (dịch đè).")
    instr = st.text_area("Yêu cầu dịch:", value="Dịch sang tiếng Việt mượt mà, văn phong chuyên nghiệp. Giữ nguyên thuật ngữ chuyên môn.")
    gloss = st.text_area("Từ điển thuật ngữ:", value="Trúc Cơ, Nguyên Anh, ROI")
    
    tab1, tab2 = st.tabs(["📄 Dịch File Đa Năng", "🌐 Dịch Link Web"])
    
    # --- TAB DỊCH FILE (NÂNG CẤP) ---
    with tab1:
        # Nâng cấp: Chấp nhận nhiều loại file
        up_files = st.file_uploader("Tải nhiều file (PDF/Docx/Ảnh/Txt):", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'jpg', 'png', 'jpeg'])
        
        if st.button("Bắt đầu dịch File"):
            if not up_files:
                st.warning("⚠️ Vui lòng chọn file trước!")
            else:
                for f in up_files:
                    st.info(f"📂 Đang xử lý file: {f.name}...")
                    
                    # 1. Dùng hàm xử lý đa năng MỚI
                    raw_contents = process_unified_file(f)
                    total_items = len(raw_contents)
                    
                    final_results = [] 
                    text_buffer = "" 
                    p_bar = st.progress(0)
                    
                    for idx, item in enumerate(raw_contents):
                        
                        # --- A. NẾU LÀ CHỮ (TEXT) ---
                        if item['type'] == 'text':
                            text_buffer += item['val'] + "\n"
                            
                            # Logic gom 2000 ký tự (giảm xuống chút để đỡ lỗi)
                            if len(text_buffer) < 2000 and idx < total_items - 1 and raw_contents[idx+1]['type'] == 'text':
                                continue
                            
                            # Dịch Text Buffer
                            try:
                                prompt_text = f"YÊU CẦU: {instr}\nTHUẬT NGỮ: {gloss}\nNỘI DUNG GỐC:\n{text_buffer}"
                                res = model.generate_content(prompt_text, safety_settings=safety_settings)
                                final_results.append({"type": "text", "val": res.text if res else text_buffer})
                            except: 
                                final_results.append({"type": "text", "val": text_buffer}) # Fallback
                            text_buffer = ""
                        
                        # --- B. NẾU LÀ ẢNH (IMAGE - LOGIC MỚI) ---
                        elif item['type'] == 'image':
                            # Dịch nốt text tồn đọng
                            if text_buffer:
                                try:
                                    res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                                    final_results.append({"type": "text", "val": res.text})
                                except: pass
                                text_buffer = ""

                            # PROMPT PHÂN LOẠI & DỊCH
                            img_prompt = [
                                f"""
                                Bạn là chuyên gia OCR & Dịch Hán Nôm. Hãy nhìn ảnh và quyết định:
                                1. [MODE:TEXT]: Nếu đây là ẢNH SCAN VĂN BẢN/SÁCH CỔ (Nhiều chữ, Hán tự đọc dọc). -> Hãy OCR và dịch toàn bộ sang Tiếng Việt.
                                2. [MODE:IMG]: Nếu đây là TRANH MINH HỌA (Ít chữ, có hình vẽ). -> Hãy dịch nội dung chữ trong tranh (nếu có).
                                
                                YÊU CẦU: {instr}
                                """,
                                item['val']
                            ]
                            
                            try:
                                res = model.generate_content(img_prompt, safety_settings=safety_settings)
                                res_txt = res.text if res else ""
                                
                                if "[MODE:TEXT]" in res_txt:
                                    # Chế độ Sách Scan -> Chuyển thành Text
                                    clean_txt = res_txt.replace("[MODE:TEXT]", "").strip()
                                    final_results.append({
                                        "type": "text", 
                                        "val": f"\n--- [Nội dung từ ảnh scan: {item.get('name')}] ---\n{clean_txt}\n",
                                        "mode_tag": "[MODE:TEXT]"
                                    })
                                elif "[MODE:IMG]" in res_txt:
                                    # Chế độ Tranh -> Dịch đè
                                    caption = res_txt.replace("[MODE:IMG]", "").strip()
                                    if caption:
                                        new_img = overlay_text_on_image(item['val'], caption)
                                        final_results.append({"type": "image", "val": item['val'], "val_translated": new_img})
                                    else:
                                        final_results.append(item) # Giữ ảnh gốc
                                else:
                                    # Không rõ -> Coi là text
                                    final_results.append({"type": "text", "val": res_txt})
                            except: 
                                final_results.append(item) # Lỗi thì giữ ảnh gốc

                        p_bar.progress((idx+1)/total_items)

                    # XỬ LÝ NỐT BUFFER CUỐI
                    if text_buffer:
                         try:
                            res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                            final_results.append({"type": "text", "val": res.text})
                         except: pass

                    st.success(f"✅ Hoàn tất file: {f.name}")
                    st.download_button(f"⬇️ Tải bản dịch {f.name}", save_docx_mixed(final_results).getvalue(), f"VN_{f.name}.docx")

    # --- TAB DỊCH WEB (GIỮ NGUYÊN) ---
    with tab2:
        urls = st.text_area("Dán danh sách Link (mỗi dòng 1 link):")
        if st.button("Bắt đầu dịch Link"):
            links = urls.split("\n")
            all_txt = []
            for l in links:
                if l.strip():
                    raw = scrape_url(l.strip())
                    if raw:
                        try:
                            res = model.generate_content(f"Dịch bài này sang tiếng Việt:\n{raw[:15000]}", safety_settings=safety_settings)
                            if res and res.text:
                                all_txt.append({"type": "text", "val": f"\n--- {l} ---\n{res.text}\n"})
                        except Exception as e:
                            all_txt.append({"type": "text", "val": f"\n[Lỗi dịch link {l}: {e}]\n"})
            st.download_button("Tải file dịch Web", save_docx_mixed(all_txt).getvalue(), "Dich_Web.docx")

# ==============================================================================
# 3. DỊCH ẢNH (OCR) - GIỮ NGUYÊN GIAO DIỆN
# ==============================================================================
elif menu == "🖼️ Dịch Ảnh (OCR)":
    st.subheader("🖼️ Dịch chữ từ Hình ảnh (Có vẽ đè)")
    imgs = st.file_uploader("Tải ảnh lên (PNG/JPG):", accept_multiple_files=True)
    
    if imgs and st.button("Bắt đầu dịch ảnh"):
        full_ocr = []
        for im_f in imgs:
            try:
                img = Image.open(im_f)
                col1, col2 = st.columns(2)
                col1.image(img, caption=f"Gốc: {im_f.name}")
                
                with st.spinner("Đang soi chữ và dịch..."):
                    res = model.generate_content(
                        ["Trích xuất chữ và dịch sang Tiếng Việt (ngắn gọn):", img], 
                        safety_settings=safety_settings
                    )
                    if res and res.text:
                        # Tự động vẽ đè để demo tính năng mới
                        new_img = overlay_text_on_image(img, res.text)
                        col2.image(new_img, caption="Đã dịch đè")
                        
                        st.write(res.text)
                        full_ocr.append({"type": "text", "val": f"\n--- {im_f.name} ---\n{res.text}\n"})
            except Exception as e:
                st.error(f"Lỗi ảnh {im_f.name}: {e}")
        
        if full_ocr:
            st.download_button("Tải file kết quả", save_docx_mixed(full_ocr).getvalue(), "Dich_Anh.docx")
