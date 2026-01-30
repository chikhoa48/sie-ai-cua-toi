import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time, textwrap
from PIL import Image, ImageDraw, ImageFont, ImageOps
import fitz  # Thư viện PyMuPDF (xử lý PDF)
from docx import Document
from docx.shared import Inches, Pt
from bs4 import BeautifulSoup

# ==============================================================================
# 1. CẤU HÌNH TRANG & API
# ==============================================================================
st.set_page_config(page_title="Siêu AI Dịch Thuật Đa Năng", page_icon="🚀", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white; border-radius: 5px; font-weight: bold;}</style>""", unsafe_allow_html=True)

# Cấu hình an toàn cho Gemini (Tránh bị chặn khi dịch văn bản cổ/nhạy cảm)
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Kết nối API Key
try:
    # Ưu tiên lấy từ Secrets của Streamlit
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        # Fallback nếu chạy local mà chưa set env
        st.warning("⚠️ Chưa tìm thấy API Key trong Secrets. Vui lòng kiểm tra file .streamlit/secrets.toml")
        st.stop()
        
    genai.configure(api_key=api_key)
    
    # Lấy danh sách model, ưu tiên Flash cho nhanh, Pro cho thông minh
    available_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
except Exception as e:
    st.error(f"⚠️ Lỗi kết nối API: {e}")
    st.stop()

# ==============================================================================
# 2. CÁC HÀM HỖ TRỢ XỬ LÝ ẢNH & FONT
# ==============================================================================
def get_font(size):
    """
    Tìm font hỗ trợ tiếng Việt trên server (Linux/Windows/Mac).
    Rất quan trọng để vẽ chữ lên ảnh không bị lỗi ô vuông.
    """
    font_paths = [
        "arial.ttf", "Arial.ttf", "Calibri.ttf", # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Streamlit Cloud / Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc" # Mac
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except: continue
    return ImageFont.load_default() # Fallback cuối cùng

def overlay_text_on_image(original_img, text_content):
    """
    Vẽ chữ tiếng Việt đè lên ảnh gốc (Dành cho tranh minh họa).
    Tạo lớp mờ đen (Overlay) để chữ nổi bật.
    """
    try:
        img = original_img.convert("RGBA")
        width, height = img.size
        
        # 1. Tạo lớp phủ mờ màu đen
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 160)) 
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")

        draw = ImageDraw.Draw(img)
        
        # 2. Tính cỡ chữ dynamic theo chiều rộng ảnh
        fontsize = int(width / 30) 
        if fontsize < 12: fontsize = 12
        font = get_font(fontsize)
        
        # 3. Ngắt dòng văn bản
        avg_char_width = fontsize * 0.6
        chars_per_line = int((width - 40) / avg_char_width)
        wrapper = textwrap.TextWrapper(width=chars_per_line)
        lines = []
        for line in text_content.split('\n'):
            lines.extend(wrapper.wrap(line))
            
        # 4. Vẽ chữ căn giữa
        text_height = len(lines) * (fontsize + 8)
        current_y = (height - text_height) / 2
        if current_y < 20: current_y = 20 # Padding top tối thiểu

        for line in lines:
            # Tính toán vị trí x để căn giữa
            try:
                left, top, right, bottom = font.getbbox(line)
                text_w = right - left
            except: text_w = len(line) * fontsize * 0.5 
            
            x_pos = (width - text_w) / 2
            if x_pos < 10: x_pos = 10

            # Vẽ viền chữ đen (shadow) cho dễ đọc
            draw.text((x_pos+2, current_y+2), line, font=font, fill="black")
            # Vẽ chữ chính màu Vàng
            draw.text((x_pos, current_y), line, font=font, fill=(255, 255, 100))
            
            current_y += fontsize + 8
            
        return img
    except Exception as e:
        print(f"Lỗi vẽ ảnh: {e}")
        return original_img

def save_docx_layout(contents):
    """Lưu danh sách nội dung (Text/Image) vào file Word"""
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
            # Kiểm tra xem lấy ảnh gốc hay ảnh đã vẽ đè
            img_to_save = item.get('val_translated', item['val']) 
            try:
                img_byte = io.BytesIO()
                img_to_save.save(img_byte, format='PNG')
                doc.add_picture(img_byte, width=Inches(5.5))
                
                # (Tuỳ chọn) Thêm chú thích dưới ảnh
                # if 'trans_text' in item and "[MODE:IMG]" in item.get('mode_tag', ''):
                #     p = doc.add_paragraph(f"[Nội dung tranh]: {item['trans_text']}")
                #     p.italic = True
            except: pass
            
    bio = io.BytesIO()
    doc.save(bio)
    return bio

# ==============================================================================
# 3. BỘ XỬ LÝ FILE ĐA NĂNG (UNIFIED FILE PROCESSOR)
# ==============================================================================

def process_pdf_layout_preserved(file_stream):
    """Xử lý riêng cho PDF: Giữ layout Text và Image theo thứ tự"""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    content_list = [] 
    for page_num, page in enumerate(doc):
        # sort=True: Đọc từ trên xuống dưới, trái sang phải
        blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            if block["type"] == 0: # Block Text
                text = ""
                for line in block["lines"]:
                    for span in line["spans"]: text += span["text"] + " "
                    text += "\n"
                if text.strip():
                    content_list.append({"type": "text", "val": text})
            elif block["type"] == 1: # Block Image
                try:
                    # Bỏ qua ảnh quá nhỏ (icon, đường kẻ)
                    if len(block["image"]) > 3000:
                        img_pil = Image.open(io.BytesIO(block["image"]))
                        content_list.append({"type": "image", "val": img_pil, "name": f"PDF_P{page_num}"})
                except: pass
    return content_list

def process_docx_with_images(file_stream):
    """
    Xử lý riêng cho DOCX: Lấy Text và Ảnh (nhúng trong XML) theo thứ tự.
    """
    doc = Document(file_stream)
    content_list = []
    
    for para in doc.paragraphs:
        # 1. Lấy Text của đoạn văn
        text = para.text
        if text.strip():
            content_list.append({"type": "text", "val": text + "\n"})
            
        # 2. "Đào" XML để tìm ảnh (Blip) gắn liền với đoạn văn này
        try:
            nsmap = para._element.nsmap
            blips = para._element.findall('.//a:blip', namespaces=nsmap)
            for blip in blips:
                embed_attr = blip.get(f"{{{nsmap['r']}}}embed") 
                if embed_attr:
                    image_part = doc.part.related_parts[embed_attr]
                    image_bytes = image_part.blob
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    content_list.append({"type": "image", "val": img_pil, "name": "DOCX_Img"})
        except: pass
            
    return content_list

def process_unified_file(uploaded_file):
    """
    ROUTER TRUNG TÂM: Phân loại file và gọi hàm xử lý tương ứng
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    content_list = []

    # 1. PDF
    if file_type == 'pdf':
        return process_pdf_layout_preserved(uploaded_file)

    # 2. WORD (DOCX)
    elif file_type == 'docx':
        return process_docx_with_images(uploaded_file)

    # 3. ẢNH (JPG, PNG,...) -> Đóng gói thành 1 khối Image
    elif file_type in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
        try:
            img = Image.open(uploaded_file)
            img = ImageOps.exif_transpose(img) # Xoay đúng chiều ảnh chụp đt
            content_list.append({"type": "image", "val": img, "name": uploaded_file.name})
        except: pass

    # 4. TEXT (TXT)
    elif file_type == 'txt':
        try:
            text = uploaded_file.getvalue().decode("utf-8")
            content_list.append({"type": "text", "val": text})
        except: pass
        
    return content_list

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (STREAMLIT UI)
# ==============================================================================
st.title("🚀 Siêu AI: Dịch Thuật & Số Hóa Đa Năng")

with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.info("ℹ️ **Mẹo:** Flash xử lý nhanh. Pro xử lý ảnh Hán Nôm/chữ viết tay tốt hơn.")
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["🏭 Dịch Tài Liệu Đa Năng", "🔮 Hỏi Đáp Chuyên Sâu"])

model = genai.GenerativeModel(selected_model)

# ------------------------------------------------------------------------------
# MODE 1: DỊCH TÀI LIỆU (SMART HYBRID MODE)
# ------------------------------------------------------------------------------
if menu == "🏭 Dịch Tài Liệu Đa Năng":
    st.subheader("🏭 Dịch Thuật (PDF - Word - Ảnh - Text)")
    st.markdown("""
    **Cơ chế Xử lý Thông minh:**
    1.  📄 **Văn bản:** Dịch giữ nguyên định dạng.
    2.  🖼️ **Ảnh Minh Họa (Truyện tranh):** Dịch và **vẽ chữ đè lên ảnh**.
    3.  📚 **Ảnh Scan (Sách Hán Nôm/Văn bản):** Tự động nhận diện, **OCR thành văn bản (Text)** để dễ đọc.
    """)
    
    instr = st.text_area("Yêu cầu dịch:", value="Dịch sang tiếng Việt văn phong kiếm hiệp, trang trọng. Giữ nguyên tên riêng và thuật ngữ Hán Việt.", height=80)
    
    # Cho phép chọn nhiều loại file
    up_files = st.file_uploader("Tải file (Chọn nhiều file cùng lúc):", 
                                accept_multiple_files=True, 
                                type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'webp'])
    
    if st.button("🚀 Bắt đầu Xử lý"):
        if not up_files:
            st.warning("⚠️ Vui lòng chọn ít nhất 1 file!")
        else:
            for f in up_files:
                st.write(f"---")
                st.write(f"📂 **Đang xử lý file:** `{f.name}`...")
                
                # BƯỚC 1: ĐỒNG BỘ HÓA ĐẦU VÀO
                try:
                    raw_contents = process_unified_file(f)
                except Exception as e:
                    st.error(f"Lỗi đọc file {f.name}: {e}")
                    continue

                total_blocks = len(raw_contents)
                if total_blocks == 0:
                    st.warning(f"File {f.name} không có nội dung đọc được.")
                    continue
                    
                st.caption(f"👉 Tìm thấy {total_blocks} phần nội dung (Text/Image).")
                
                final_results = []
                text_buffer = ""
                
                p_bar = st.progress(0)
                status_text = st.empty()

                for idx, item in enumerate(raw_contents):
                    status_text.text(f"Đang AI xử lý phần {idx+1}/{total_blocks}...")
                    
                    # === TRƯỜNG HỢP A: TEXT ===
                    if item['type'] == 'text':
                        text_buffer += item['val'] + "\n"
                        # Gom 3000 ký tự rồi dịch 1 lần để tiết kiệm request
                        if len(text_buffer) < 3000 and idx < total_blocks - 1 and raw_contents[idx+1]['type'] == 'text':
                            continue
                        
                        if text_buffer.strip():
                            try:
                                prompt = f"Dịch đoạn văn bản sau sang Tiếng Việt. YÊU CẦU: {instr}\n\nNỘI DUNG GỐC:\n{text_buffer}"
                                res = model.generate_content(prompt, safety_settings=safety_settings)
                                final_results.append({"type": "text", "val": res.text if res else text_buffer})
                            except: 
                                final_results.append({"type": "text", "val": text_buffer}) # Fallback
                            text_buffer = ""

                    # === TRƯỜNG HỢP B: IMAGE (PDF, DOCX, JPG...) ===
                    elif item['type'] == 'image':
                        # Dịch nốt text buffer tồn đọng
                        if text_buffer.strip():
                            try:
                                res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                                final_results.append({"type": "text", "val": res.text})
                            except: pass
                            text_buffer = ""

                        # --- PROMPT PHÂN LOẠI & DỊCH ---
                        img_prompt = [
                            f"""
                            Bạn là chuyên gia OCR và Dịch thuật Hán Nôm/Cổ văn. Hãy nhìn ảnh và thực hiện:
                            
                            1. [PHÂN LOẠI]:
                               - Nếu đây là **Tranh minh họa, Bìa sách** (có hình vẽ nhân vật/cảnh, ít chữ): Quyết định là chế độ ẢNH.
                               - Nếu đây là **Ảnh Scan văn bản, Trang sách cổ** (chứa nhiều chữ, văn bản hành chính, sách Hán Nôm): Quyết định là chế độ TEXT.
                            
                            2. [THỰC HIỆN]:
                               - Nếu chế độ TEXT: Hãy OCR toàn bộ chữ (Đọc DỌC từ Phải->Trái nếu là Hán cổ). Dịch sang Tiếng Việt. Trả về bắt đầu bằng `[MODE:TEXT]`.
                               - Nếu chế độ ẢNH: Chỉ dịch nội dung chữ trong tranh (nếu có). Trả về bắt đầu bằng `[MODE:IMG]`.
                            
                            YÊU CẦU DỊCH: {instr}
                            """,
                            item['val']
                        ]
                        
                        try:
                            # Gọi Gemini Vision
                            res_img = model.generate_content(img_prompt, safety_settings=safety_settings)
                            response_content = res_img.text if res_img else ""
                            
                            if "[MODE:TEXT]" in response_content:
                                # ==> ẢNH SCAN -> CHUYỂN THÀNH TEXT (Bỏ ảnh)
                                clean_text = response_content.replace("[MODE:TEXT]", "").strip()
                                final_results.append({
                                    "type": "text", 
                                    "val": f"\n--- [Nội dung từ ảnh scan: {item.get('name')}] ---\n{clean_text}\n"
                                })
                                st.toast(f"📖 Đã chuyển đổi 1 trang sách ảnh sang Text!")

                            elif "[MODE:IMG]" in response_content:
                                # ==> TRANH MINH HỌA -> GIỮ ẢNH & DỊCH ĐÈ
                                caption = response_content.replace("[MODE:IMG]", "").strip()
                                if caption:
                                    new_img = overlay_text_on_image(item['val'], caption)
                                    final_results.append({
                                        "type": "image", 
                                        "val": item['val'],
                                        "val_translated": new_img,
                                        "trans_text": caption,
                                        "mode_tag": "[MODE:IMG]"
                                    })
                                else:
                                    final_results.append(item) # Giữ ảnh gốc nếu ko có chữ
                                st.toast(f"🖼️ Đã xử lý tranh minh hoạ!")
                                
                            else:
                                # Fallback: Nếu AI không trả thẻ MODE, coi như Text cho an toàn
                                final_results.append({"type": "text", "val": response_content})

                        except Exception as e:
                            st.error(f"Lỗi Vision AI: {e}")
                            final_results.append(item)
                            
                    p_bar.progress((idx+1)/total_blocks)
                
                # Xử lý text cuối cùng
                if text_buffer.strip():
                    try:
                        res = model.generate_content(f"Dịch: {text_buffer}", safety_settings=safety_settings)
                        final_results.append({"type": "text", "val": res.text})
                    except: pass

                st.success(f"✅ Hoàn tất file: {f.name}")
                
                # Tải file kết quả
                docx_file = save_docx_layout(final_results)
                st.download_button(
                    label=f"⬇️ Tải Word ({f.name})",
                    data=docx_file.getvalue(),
                    file_name=f"VN_{f.name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# ------------------------------------------------------------------------------
# MODE 2: HỎI ĐÁP
# ------------------------------------------------------------------------------
elif menu == "🔮 Hỏi Đáp Chuyên Sâu":
    st.subheader("🔮 Trợ Lý Hỏi Đáp & Phân Tích")
    
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "context" not in st.session_state: st.session_state.context = ""

    with st.expander("📚 Nạp Tài Liệu Bổ Sung (Tùy chọn)"):
        files = st.file_uploader("Upload tài liệu (PDF, Word, TXT) để AI học:", accept_multiple_files=True)
        if st.button("Học tài liệu"):
            if files:
                raw_text = ""
                with st.spinner("Đang đọc tài liệu..."):
                    for fl in files:
                        # Tái sử dụng hàm đọc file đa năng để lấy text
                        blocks = process_unified_file(fl)
                        for b in blocks:
                            if b['type'] == 'text': raw_text += b['val'] + "\n"
                
                st.session_state.context += raw_text
                st.success(f"Đã nạp {len(raw_text)} ký tự vào bộ nhớ tạm!")

    # Hiển thị lịch sử chat
    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).markdown(m["content"])

    if q := st.chat_input("Hỏi gì đó về tài liệu hoặc kiến thức chung..."):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.chat_message("user").markdown(q)
        
        full_prompt = f"Dựa vào KIẾN THỨC NỀN SAU (nếu có): {st.session_state.context}\n\nTrả lời câu hỏi: {q}"
        try:
            res = model.generate_content(full_prompt)
            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append({"role": "assistant", "content": res.text})
        except Exception as e:
            st.error(f"Lỗi: {e}")
