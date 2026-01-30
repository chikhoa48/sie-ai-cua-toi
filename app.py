import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time, textwrap
from PIL import Image, ImageDraw, ImageFont
import fitz  # Thư viện PyMuPDF
from docx import Document
from docx.shared import Inches, Pt
from bs4 import BeautifulSoup

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Siêu AI Dịch Thuật Đa Năng", page_icon="🚀", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white; border-radius: 5px;}</style>""", unsafe_allow_html=True)

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
    
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                available_models.append(m.name)
    except: pass
    
    if not available_models: 
        available_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
    else:
        # Ưu tiên Flash cho nhanh, Pro cho thông minh
        available_models.sort(key=lambda x: "flash" not in x)
except:
    st.error("⚠️ Chưa nhập GEMINI_API_KEY trong .streamlit/secrets.toml")
    st.stop()

# ==============================================================================
# 1. CÁC HÀM HỖ TRỢ XỬ LÝ ẢNH & FONT
# ==============================================================================
def get_font(size):
    """Tìm font hỗ trợ tiếng Việt trong hệ thống server"""
    font_paths = [
        "arial.ttf", "Arial.ttf", # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux (Streamlit Cloud)
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc" # Mac
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except: continue
    return ImageFont.load_default() 

def overlay_text_on_image(original_img, text_content):
    """
    Vẽ chữ tiếng Việt đè lên ảnh gốc (Dành cho tranh minh họa).
    Tạo lớp mờ đen để chữ nổi bật.
    """
    try:
        img = original_img.convert("RGBA")
        width, height = img.size
        
        # Tạo lớp phủ mờ màu đen (độ trong suốt 160/255)
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 160)) 
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")

        draw = ImageDraw.Draw(img)
        
        # Tính cỡ chữ dynamic theo chiều rộng ảnh
        fontsize = int(width / 28) 
        if fontsize < 14: fontsize = 14
        font = get_font(fontsize)
        
        # Ngắt dòng văn bản
        avg_char_width = fontsize * 0.6
        chars_per_line = int((width - 40) / avg_char_width)
        wrapper = textwrap.TextWrapper(width=chars_per_line)
        lines = []
        for line in text_content.split('\n'):
            lines.extend(wrapper.wrap(line))
            
        # Vẽ chữ căn giữa
        text_height = len(lines) * (fontsize + 8)
        current_y = (height - text_height) / 2
        if current_y < 20: current_y = 20

        for line in lines:
            # Lấy kích thước dòng để căn giữa
            try:
                left, top, right, bottom = font.getbbox(line)
                text_w = right - left
            except: text_w = len(line) * fontsize * 0.5 
            
            x_pos = (width - text_w) / 2
            if x_pos < 10: x_pos = 10

            # Vẽ viền chữ đen (shadow)
            draw.text((x_pos+2, current_y+2), line, font=font, fill="black")
            # Vẽ chữ chính màu Vàng chanh
            draw.text((x_pos, current_y), line, font=font, fill=(255, 255, 100))
            
            current_y += fontsize + 8
            
        return img
    except Exception as e:
        print(f"Lỗi vẽ ảnh: {e}")
        return original_img

# ==============================================================================
# 2. HÀM XỬ LÝ PDF THÔNG MINH (LAYOUT)
# ==============================================================================
def process_pdf_layout_preserved(file_stream):
    """
    Đọc PDF và trả về danh sách các Block theo đúng thứ tự hiển thị.
    """
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    content_list = [] 

    for page_num, page in enumerate(doc):
        # sort=True: Quan trọng để lấy đúng thứ tự trên -> dưới
        blocks = page.get_text("dict", sort=True)["blocks"]
        
        for block in blocks:
            # --- XỬ LÝ TEXT (Type 0) ---
            if block["type"] == 0: 
                text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                    text += "\n"
                if text.strip():
                    content_list.append({"type": "text", "val": text})

            # --- XỬ LÝ IMAGE (Type 1) ---
            elif block["type"] == 1: 
                try:
                    image_bytes = block["image"]
                    if len(image_bytes) > 5120: # Lọc ảnh rác < 5KB
                        img_pil = Image.open(io.BytesIO(image_bytes))
                        content_list.append({
                            "type": "image", 
                            "val": img_pil, 
                            "name": f"Trang{page_num+1}"
                        })
                except: pass
            
    return content_list

def save_docx_layout(contents):
    """Lưu kết quả ra file Word"""
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)

    for item in contents:
        if item['type'] == 'text':
            clean_text = item['val'].strip()
            if clean_text:
                doc.add_paragraph(clean_text)

        elif item['type'] == 'image':
            # Kiểm tra xem là ảnh gốc hay ảnh đã vẽ đè
            img_to_save = item.get('val_translated', item['val']) 
            try:
                img_byte = io.BytesIO()
                img_to_save.save(img_byte, format='PNG')
                doc.add_picture(img_byte, width=Inches(5.5))
                # Nếu muốn thêm chú thích
                # if 'trans_text' in item:
                #     p = doc.add_paragraph(f"[Minh hoạ]: {item['trans_text']}")
                #     p.italic = True
            except: pass
            
    bio = io.BytesIO()
    doc.save(bio)
    return bio

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================
st.title("🚀 Siêu Trợ Lý: Dịch Thuật & OCR Hán Nôm")

with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.caption("Mẹo: 'Pro' xử lý ảnh Hán Nôm tốt hơn 'Flash'.")
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["🏭 Dịch Tài Liệu (PDF Layout)", "🔮 Hỏi Đáp Chuyên Sâu", "🖼️ Dịch Ảnh Lẻ"])

model = genai.GenerativeModel(selected_model)

# ------------------------------------------------------------------------------
# CHỨC NĂNG 1: DỊCH TÀI LIỆU (SMART HYBRID MODE)
# ------------------------------------------------------------------------------
if menu == "🏭 Dịch Tài Liệu (PDF Layout)":
    st.subheader("🏭 Dịch PDF - Tự động nhận diện Ảnh Scan & Hán Nôm")
    st.info("""
    **Cơ chế thông minh:**
    1. **Ảnh minh họa:** Giữ nguyên ảnh, dịch đè chữ tiếng Việt lên ảnh.
    2. **Ảnh Scan (Sách cổ/Hán Nôm):** Tự động chuyển thành văn bản (Text) để dễ đọc, loại bỏ ảnh nền.
    """)
    
    instr = st.text_area("Yêu cầu dịch:", value="Dịch sang tiếng Việt văn phong kiếm hiệp, trang trọng. Giữ nguyên các thuật ngữ Hán Việt đặc thù.", height=80)
    
    up_files = st.file_uploader("Tải file PDF (Có thể chứa ảnh scan):", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Bắt đầu Dịch"):
        if not up_files:
            st.warning("⚠️ Vui lòng chọn file PDF!")
        else:
            for f in up_files:
                st.write(f"📂 **Đang xử lý file:** `{f.name}`...")
                
                # 1. Phân tích cấu trúc file
                raw_contents = process_pdf_layout_preserved(f) 
                total_blocks = len(raw_contents)
                st.write(f"👉 Tìm thấy {total_blocks} khối nội dung.")
                
                final_results = []
                text_buffer = ""
                
                p_bar = st.progress(0)
                status_text = st.empty()

                for idx, item in enumerate(raw_contents):
                    status_text.caption(f"Đang xử lý khối {idx+1}/{total_blocks} ({item['type']})...")
                    
                    # --- XỬ LÝ TEXT ---
                    if item['type'] == 'text':
                        text_buffer += item['val'] + "\n"
                        # Gom khoảng 3000 ký tự rồi dịch 1 lần
                        if len(text_buffer) < 3000 and idx < total_blocks - 1 and raw_contents[idx+1]['type'] == 'text':
                            continue
                        
                        if text_buffer.strip():
                            res_text = ""
                            try:
                                prompt = f"Dịch đoạn văn bản sau sang Tiếng Việt. YÊU CẦU: {instr}\n\nNỘI DUNG:\n{text_buffer}"
                                res = model.generate_content(prompt, safety_settings=safety_settings)
                                if res and res.text: res_text = res.text
                            except: res_text = text_buffer
                            
                            final_results.append({"type": "text", "val": res_text})
                            text_buffer = ""

                    # --- XỬ LÝ IMAGE (HYBRID LOGIC) ---
                    elif item['type'] == 'image':
                        # Dịch nốt text tồn đọng trước khi xử lý ảnh
                        if text_buffer.strip():
                            try:
                                res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                                final_results.append({"type": "text", "val": res.text})
                            except: pass
                            text_buffer = ""

                        # PROMPT THÔNG MINH: PHÂN LOẠI & DỊCH
                        img_prompt = [
                            f"""
                            Bạn là chuyên gia Hán Nôm & OCR. Hãy phân tích hình ảnh này:
                            
                            1. Nếu là **Tranh minh họa** (ít chữ, có hình vẽ nhân vật/cảnh): 
                               - Dịch nội dung chữ trong tranh (nếu có).
                               - Trả về kết quả bắt đầu bằng: `[MODE:IMG]` theo sau là nội dung dịch.
                               
                            2. Nếu là **Ảnh Scan văn bản/Trang sách cổ** (chứa nhiều chữ, Hán văn cổ):
                               - Đọc toàn bộ chữ (Lưu ý: Hán cổ đọc DỌC từ Phải -> Trái, Trên -> Dưới).
                               - Dịch toàn bộ sang Tiếng Việt hiện đại, chia đoạn rõ ràng.
                               - Trả về kết quả bắt đầu bằng: `[MODE:TEXT]` theo sau là nội dung dịch.
                            
                            YÊU CẦU DỊCH: {instr}
                            """,
                            item['val']
                        ]
                        
                        try:
                            # Gọi AI Vision
                            res_img = model.generate_content(img_prompt, safety_settings=safety_settings)
                            response_content = res_img.text if res_img else ""
                            
                            if "[MODE:TEXT]" in response_content:
                                # ==> ĐÂY LÀ ẢNH SÁCH SCAN -> CHUYỂN THÀNH TEXT
                                clean_text = response_content.replace("[MODE:TEXT]", "").strip()
                                final_results.append({
                                    "type": "text", 
                                    "val": f"\n[Nội dung từ trang sách ảnh - {item.get('name')}]\n{clean_text}\n"
                                })
                                st.toast(f"📖 Đã chuyển đổi 1 trang sách ảnh sang Text!")

                            elif "[MODE:IMG]" in response_content:
                                # ==> ĐÂY LÀ TRANH MINH HỌA -> VẼ ĐÈ
                                caption = response_content.replace("[MODE:IMG]", "").strip()
                                if caption:
                                    new_img = overlay_text_on_image(item['val'], caption)
                                    final_results.append({
                                        "type": "image", 
                                        "val": item['val'],
                                        "val_translated": new_img,
                                        "trans_text": caption
                                    })
                                else:
                                    final_results.append(item) # Giữ ảnh gốc nếu không có chữ
                                st.toast(f"🖼️ Đã dịch và vẽ đè 1 tranh minh hoạ!")
                                
                            else:
                                # Fallback: Nếu AI không phân loại được, coi là Text cho an toàn
                                final_results.append({"type": "text", "val": response_content})

                        except Exception as e:
                            st.error(f"Lỗi Vision AI: {e}")
                            final_results.append(item) # Giữ nguyên ảnh gốc nếu lỗi
                            
                    p_bar.progress((idx+1)/total_blocks)
                
                # Xử lý text buffer cuối cùng
                if text_buffer.strip():
                    try:
                        res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                        final_results.append({"type": "text", "val": res.text})
                    except: pass

                st.success(f"✅ Hoàn tất file: {f.name}")
                
                # Tạo file download
                docx_file = save_docx_layout(final_results)
                st.download_button(
                    label=f"⬇️ Tải bản dịch Word ({f.name})",
                    data=docx_file.getvalue(),
                    file_name=f"VN_Full_{f.name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# ------------------------------------------------------------------------------
# CHỨC NĂNG 2: HỎI ĐÁP
# ------------------------------------------------------------------------------
elif menu == "🔮 Hỏi Đáp Chuyên Sâu":
    st.subheader("🔮 Trợ Lý Hỏi Đáp (Huyền học/Data)")
    
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "context" not in st.session_state: st.session_state.context = ""

    with st.expander("📚 Nạp kiến thức bổ sung (Tùy chọn)"):
        files = st.file_uploader("Nạp tài liệu PDF/TXT:", accept_multiple_files=True)
        if st.button("Học tài liệu"):
            if files:
                raw_text = ""
                for fl in files:
                    if fl.name.endswith(".pdf"):
                        with fitz.open(stream=fl.read(), filetype="pdf") as doc:
                            for p in doc: raw_text += p.get_text()
                    else:
                        raw_text += fl.getvalue().decode("utf-8")
                st.session_state.context = raw_text
                st.success("Đã nạp xong kiến thức!")

    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).markdown(m["content"])

    if q := st.chat_input("Nhập câu hỏi..."):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        
        full_prompt = f"KIẾN THỨC NỀN: {st.session_state.context}\n\nCÂU HỎI: {q}"
        try:
            res = model.generate_content(full_prompt)
            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append({"role": "assistant", "content": res.text})
        except Exception as e:
            st.error(f"Lỗi: {e}")

# ------------------------------------------------------------------------------
# CHỨC NĂNG 3: DỊCH ẢNH LẺ
# ------------------------------------------------------------------------------
elif menu == "🖼️ Dịch Ảnh Lẻ":
    st.subheader("🖼️ Dịch Ảnh Nhanh (OCR & Overlay)")
    uploaded_files = st.file_uploader("Tải ảnh (PNG/JPG):", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh Gốc", use_container_width=True)
            
            if st.button(f"Dịch ảnh: {f.name}"):
                with st.spinner("Đang AI xử lý..."):
                    try:
                        res = model.generate_content(["Dịch nội dung trong ảnh sang tiếng Việt (Giữ ngắn gọn):", img])
                        if res and res.text:
                            new_img = overlay_text_on_image(img, res.text)
                            with col2:
                                st.image(new_img, caption="Ảnh Dịch", use_container_width=True)
                            st.success("Nội dung text:\n" + res.text)
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
