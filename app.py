import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, io, requests, time
from PIL import Image
# Thư viện xử lý PDF mạnh mẽ (Lấy cả ảnh và chữ)
import fitz  
from docx import Document
from docx.shared import Inches
from bs4 import BeautifulSoup

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Siêu AI Đa Năng", page_icon="🚀", layout="wide")
st.markdown("""<style>.stButton>button {background-color: #d35400; color: white;}</style>""", unsafe_allow_html=True)

# --- CẤU HÌNH AN TOÀN (GIỮ NGUYÊN NHƯ YÊU CẦU) ---
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
    
    # Logic chọn model giữ nguyên
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                available_models.append(m.name)
    except: pass
    
    if not available_models: 
        available_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
    else:
        available_models.sort(key=lambda x: "flash" not in x)
except:
    st.error("⚠️ Chưa nhập API Key trong Secrets.")
    st.stop()

# --- HÀM XỬ LÝ FILE MỚI (HỖ TRỢ TÁCH ẢNH TỪ PDF) ---
def process_pdf_mixed(file_stream):
    """
    Hàm này đọc PDF từng trang:
    - Lấy chữ (Text) gom vào chuỗi.
    - Lấy ảnh (Image) lưu vào danh sách kèm vị trí trang.
    """
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    content_list = [] # Danh sách chứa các cục (Text hoặc Image) theo thứ tự

    for page_num, page in enumerate(doc):
        # 1. Lấy Chữ của trang đó
        text = page.get_text()
        if text.strip():
            content_list.append({"type": "text", "val": text})

        # 2. Lấy Ảnh của trang đó
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                # Bỏ qua ảnh quá nhỏ (icon, đường kẻ) < 5KB
                if len(image_bytes) > 5120:
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    content_list.append({"type": "image", "val": img_pil, "name": f"Trang_{page_num+1}_Anh_{img_index+1}"})
            except: pass
            
    return content_list

def get_text_only(files):
    # Hàm cũ để dùng cho RAG (Chỉ lấy chữ)
    text = ""
    for f in files:
        try:
            if f.name.endswith('.pdf'):
                doc = fitz.open(stream=f.read(), filetype="pdf")
                for page in doc: text += page.get_text()
            elif f.name.endswith('.docx'):
                doc = Document(f)
                for para in doc.paragraphs: text += para.text + "\n"
            elif f.name.endswith('.txt'):
                text += f.getvalue().decode("utf-8")
        except: pass
    return text

def save_docx_mixed(contents):
    # Hàm lưu file Word có cả ảnh và chữ
    doc = Document()
    for item in contents:
        if item['type'] == 'text':
            for line in item['val'].split('\n'):
                if line.strip(): doc.add_paragraph(line)
        elif item['type'] == 'image':
            # Chèn ảnh
            try:
                img_byte = io.BytesIO()
                item['val'].save(img_byte, format='PNG')
                doc.add_picture(img_byte, width=Inches(4.5))
                # Chèn chú thích dịch
                if 'trans' in item:
                    p = doc.add_paragraph()
                    run = p.add_run(f"\n[DỊCH ẢNH TRÊN]:\n{item['trans']}")
                    run.bold = True
                    run.italic = True
                    doc.add_paragraph("-" * 20)
            except: pass
            
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
st.title("🚀 Siêu Trợ Lý: Huyền Học - Marketing - Dịch Thuật")

with st.sidebar:
    st.header("⚙️ CẤU HÌNH")
    selected_model = st.selectbox("Chọn Model:", available_models)
    st.caption("Mẹo: Dùng 'Flash' để dịch nhanh, 'Pro' để thông minh hơn.")
    st.divider()
    menu = st.radio("CHỨC NĂNG:", ["🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)", "🏭 Dịch Thuật Công Nghiệp", "🖼️ Dịch Ảnh (OCR)"])

model = genai.GenerativeModel(selected_model)

# ==============================================================================
# 1. HỎI ĐÁP CHUYÊN SÂU
# ==============================================================================
if menu == "🔮 Hỏi Đáp Chuyên Sâu (Huyền học/Data)":
    st.subheader("🔮 Trợ Lý Chuyên Gia")
    
    with st.sidebar:
        role = st.selectbox("Vai trò AI:", ["Đại sư Huyền học (Giang Công)", "Chuyên gia Marketing & Data", "Trợ lý đa năng"])
        files = st.file_uploader("Nạp tài liệu (PDF/Docx):", accept_multiple_files=True)
        if st.button("Nạp vào bộ não"):
            if files:
                st.session_state.context = get_text_only(files)
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
                else:
                    st.error("AI không trả lời được câu này.")
            except Exception as e: st.error(f"Lỗi: {e}")

# ==============================================================================
# 2. DỊCH THUẬT CÔNG NGHIỆP (LOGIC MỚI: PDF MIXED)
# ==============================================================================
elif menu == "🏭 Dịch Thuật Công Nghiệp":
    st.subheader("🏭 Dịch Sách & Truyện Hàng Loạt (Hỗ trợ Ảnh & Hán Nôm)")
    
    # --- CẤU HÌNH PROMPT ---
    st.markdown("ℹ️ **Cơ chế:** Tự động tách chữ và ảnh. Nếu gặp ảnh Hán Nôm, AI sẽ tự xoay chiều đọc từ Phải qua Trái, Trên xuống Dưới.")
    instr = st.text_area("Yêu cầu dịch:", value="Dịch sang tiếng Việt mượt mà, văn phong chuyên nghiệp. Giữ nguyên thuật ngữ chuyên môn.")
    gloss = st.text_area("Từ điển thuật ngữ:", value="Trúc Cơ, Nguyên Anh, ROI")
    
    tab1, tab2 = st.tabs(["📄 Dịch File (PDF/Docx)", "🌐 Dịch Link Web"])
    
    # --- TAB DỊCH FILE ---
    with tab1:
        up_files = st.file_uploader("Tải nhiều file:", accept_multiple_files=True)
        
        if st.button("Bắt đầu dịch File"):
            if not up_files:
                st.warning("⚠️ Vui lòng chọn file trước!")
            else:
                for f in up_files:
                    st.info(f"📂 Đang xử lý file: {f.name}...")
                    
                    final_results = [] # Chứa kết quả cuối cùng để ghi vào Word
                    
                    # 1. PHÂN TÍCH FILE (TÁCH ẢNH & CHỮ)
                    raw_contents = []
                    if f.name.endswith('.pdf'):
                        try:
                            raw_contents = process_pdf_mixed(f)
                            st.write(f"👉 Tìm thấy: {len([x for x in raw_contents if x['type']=='text'])} đoạn chữ và {len([x for x in raw_contents if x['type']=='image'])} hình ảnh.")
                        except Exception as e:
                            st.error(f"Lỗi đọc PDF (cần cài pymupdf): {e}")
                            continue
                    else:
                        # Các file khác xử lý như cũ (chỉ lấy chữ)
                        txt = get_text_only([f])
                        if txt: raw_contents = [{"type": "text", "val": txt}]

                    # 2. XỬ LÝ DỊCH (GIỮ NGUYÊN LOGIC 20000 KÝ TỰ CỦA BẠN)
                    text_buffer = "" # Bộ đệm để gom chữ cho đủ 20k
                    
                    p_bar = st.progress(0)
                    total_items = len(raw_contents)
                    
                    for idx, item in enumerate(raw_contents):
                        
                        # --- NẾU LÀ CHỮ (TEXT) ---
                        if item['type'] == 'text':
                            text_buffer += item['val'] + "\n"
                            
                            # Nếu bộ đệm chưa đủ 20.000 ký tự và chưa phải cuối cùng -> Gom tiếp
                            if len(text_buffer) < 20000 and idx < total_items - 1:
                                continue
                            
                            # Nếu đủ 20.000 hoặc đã hết file -> Dịch ngay
                            current_chunk = text_buffer
                            text_buffer = "" # Reset bộ đệm
                            
                            # == LOGIC THỬ LẠI (GIỮ NGUYÊN CỦA BẠN) ==
                            res_text = ""
                            for attempt in range(3):
                                try:
                                    prompt_text = f"YÊU CẦU: {instr}\nTHUẬT NGỮ: {gloss}\nNỘI DUNG GỐC:\n{current_chunk}"
                                    res = model.generate_content(prompt_text, safety_settings=safety_settings)
                                    if res and res.text:
                                        res_text = res.text
                                        break
                                except Exception as e:
                                    if "ResourceExhausted" in str(e):
                                        st.toast(f"⏳ Mạng bận, đợi 20s...")
                                        time.sleep(20)
                                    else:
                                        time.sleep(2)
                            
                            if res_text:
                                final_results.append({"type": "text", "val": res_text})
                            else:
                                final_results.append({"type": "text", "val": "\n[Đoạn này bị lỗi không dịch được]\n"})
                        
                        # --- NẾU LÀ ẢNH (IMAGE) ---
                        elif item['type'] == 'image':
                            # Nếu còn chữ tồn đọng trong buffer -> Dịch nốt trước khi xử lý ảnh
                            if text_buffer:
                                # (Copy y hệt logic dịch chữ ở trên)
                                current_chunk = text_buffer
                                text_buffer = ""
                                res_text_buf = ""
                                for attempt in range(3):
                                    try:
                                        prompt_text = f"YÊU CẦU: {instr}\nTHUẬT NGỮ: {gloss}\nNỘI DUNG GỐC:\n{current_chunk}"
                                        res = model.generate_content(prompt_text, safety_settings=safety_settings)
                                        if res and res.text:
                                            res_text_buf = res.text
                                            break
                                    except: time.sleep(5)
                                if res_text_buf: final_results.append({"type": "text", "val": res_text_buf})

                            # DỊCH ẢNH (DÙNG PROMPT HÁN NÔM ĐẶC BIỆT)
                            img_trans = ""
                            img_prompt = [
                                f"""
                                Hãy phân tích hình ảnh này và dịch toàn bộ chữ trong ảnh sang Tiếng Việt.
                                QUY TẮC QUAN TRỌNG:
                                1. Nếu đây là trang sách Hán Nôm cổ: Chữ thường viết DỌC từ PHẢI SANG TRÁI. Hãy đọc theo đúng thứ tự đó.
                                2. Dịch nghĩa sang tiếng Việt hiện đại (viết ngang từ Trái sang Phải).
                                3. {instr}
                                """,
                                item['val']
                            ]
                            
                            for attempt in range(3):
                                try:
                                    res = model.generate_content(img_prompt, safety_settings=safety_settings)
                                    if res and res.text:
                                        img_trans = res.text
                                        break
                                except Exception as e:
                                    if "ResourceExhausted" in str(e): time.sleep(20)
                                    else: time.sleep(2)
                            
                            # Lưu kết quả ảnh + bản dịch
                            final_results.append({"type": "image", "val": item['val'], "trans": img_trans})
                            st.toast(f"📸 Đã dịch xong 1 ảnh trong file PDF")

                        p_bar.progress((idx+1)/total_items)

                    # XỬ LÝ NỐT BUFFER CUỐI CÙNG (NẾU CÒN)
                    if text_buffer:
                         res_text = ""
                         for attempt in range(3):
                            try:
                                prompt_text = f"YÊU CẦU: {instr}\nTHUẬT NGỮ: {gloss}\nNỘI DUNG GỐC:\n{text_buffer}"
                                res = model.generate_content(prompt_text, safety_settings=safety_settings)
                                if res and res.text: res_text = res.text; break
                            except: time.sleep(5)
                         if res_text: final_results.append({"type": "text", "val": res_text})

                    st.success(f"✅ Hoàn tất file: {f.name}")
                    st.download_button(f"⬇️ Tải bản dịch {f.name}", save_docx_mixed(final_results).getvalue(), f"VN_{f.name}.docx")

    # --- TAB DỊCH WEB ---
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
# 3. DỊCH ẢNH (OCR)
# ==============================================================================
elif menu == "🖼️ Dịch Ảnh (OCR)":
    st.subheader("🖼️ Dịch chữ từ Hình ảnh")
    imgs = st.file_uploader("Tải ảnh lên (PNG/JPG):", accept_multiple_files=True)
    
    if imgs and st.button("Bắt đầu dịch ảnh"):
        full_ocr = []
        for im_f in imgs:
            try:
                img = Image.open(im_f)
                st.image(img, caption=f"Ảnh: {im_f.name}", width=300)
                
                with st.spinner("Đang soi chữ và dịch..."):
                    res = model.generate_content(
                        ["Trích xuất toàn bộ chữ trong ảnh (Ưu tiên đọc dọc phải-trái nếu là Hán cổ) và dịch sang Tiếng Việt:", img], 
                        safety_settings=safety_settings
                    )
                    if res and res.text:
                        st.write(res.text)
                        full_ocr.append({"type": "text", "val": f"\n--- {im_f.name} ---\n{res.text}\n"})
                    else:
                        st.warning(f"Không đọc được ảnh {im_f.name}")
            except Exception as e:
                st.error(f"Lỗi ảnh {im_f.name}: {e}")
        
        if full_ocr:
            st.download_button("Tải file kết quả", save_docx_mixed(full_ocr).getvalue(), "Dich_Anh.docx")
