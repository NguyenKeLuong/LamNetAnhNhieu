import streamlit as st
import torch
import os
from PIL import Image
import numpy as np
import io
import torchvision.transforms as transforms
from model import load_model, process_image, calculate_metrics

# Thiết lập tiêu đề trang
st.set_page_config(
    page_title="Hệ thống Làm Nét Ảnh Y Tế",
    page_icon="🏥",
    layout="wide"
)

# CSS để tùy chỉnh giao diện
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #3498DB, #16A085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    .sub-title {
        font-size: 24px;
        font-weight: bold;
        color: #3498DB;
        margin-top: 15px;
        margin-bottom: 10px;
        border-bottom: 2px solid #eaeaea;
        padding-bottom: 5px;
    }
    .metric-box {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: #16A085;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
        position: relative;
    }
    .image-preview {
        max-height: 300px;
        width: auto;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .image-preview:hover {
        transform: scale(1.02);
    }
    .image-fullsize {
        width: 100%;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #3498DB;
        padding: 12px 20px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .upload-section {
        background-color: #f9fafb;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #7f8c8d;
        font-size: 14px;
        margin-top: 30px;
        border-top: 1px solid #eaeaea;
    }
    .download-btn {
        background-color: #16A085;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        display: block;
        transition: background-color 0.3s ease;
    }
    .download-btn:hover {
        background-color: #138D75;
    }
    .tab-selected {
        background-color: #3498DB !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
    .modal-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    .modal-content {
        max-width: 90%;
        max-height: 90%;
    }
    .close-button {
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 30px;
        color: white;
        cursor: pointer;
    }
    .loader {
        border: 5px solid #f3f3f3;
        border-radius: 50%;
        border-top: 5px solid #3498db;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .view-button {
        width: 100%;
        margin-top: 8px;
    }
    .center-text {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown('<div class="main-title">Hệ Thống Làm Nét Ảnh Y Tế</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Sử dụng mô hình học sâu dựa trên kiến trúc mạng sinh đối kháng (GAN) để làm nét và nâng cao chất lượng ảnh y tế. Công cụ này giúp cải thiện độ phân giải của ảnh X-quang, CT, MRI và các hình ảnh y tế khác.</div>', unsafe_allow_html=True)

# Đường dẫn đến file mô hình
@st.cache_resource
def get_model():
    """
    Tải mô hình và lưu vào cache để tránh tải lại mỗi khi refresh trang.
    """
    # Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đường dẫn đến file mô hình (thay đổi theo môi trường của bạn)
    model_path = 'netG_4x_epoch1.pth (2).tar'
    
    # Kiểm tra xem file mô hình có tồn tại không
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy file mô hình tại đường dẫn: {model_path}")
        st.stop()
    
    # Tải mô hình
    try:
        model = load_model(model_path, device)
        return model, device
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()

# Khởi tạo mô hình
model, device = get_model()

# Khởi tạo session state để lưu trạng thái của ứng dụng
if 'show_fullsize_original' not in st.session_state:
    st.session_state.show_fullsize_original = False
if 'show_fullsize_enhanced' not in st.session_state:
    st.session_state.show_fullsize_enhanced = False

# Tab chính
tab1, tab2 = st.tabs(["Làm nét ảnh", "Hướng dẫn sử dụng"])

with tab1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tải lên ảnh y tế để làm nét</div>', unsafe_allow_html=True)
    
    # Tải ảnh lên
    uploaded_file = st.file_uploader("Chọn ảnh y tế", type=["jpg", "jpeg", "png", "bmp"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Đọc ảnh
        image = Image.open(uploaded_file).convert('RGB')
        
        # Hiển thị ảnh gốc - phiên bản thu nhỏ (preview)
        st.markdown('<div class="sub-title">Ảnh gốc</div>', unsafe_allow_html=True)
        
        preview_col1, preview_col2 = st.columns([1, 2])
        with preview_col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="", width=250, output_format="PNG", clamp=True)
            if st.button("Xem ảnh gốc kích thước đầy đủ", key="view_original", help="Nhấp để xem ảnh ở kích thước đầy đủ"):
                st.session_state.show_fullsize_original = True
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Modal cho ảnh gốc kích thước đầy đủ
        if st.session_state.show_fullsize_original:
            with st.container():
                st.markdown('<div class="sub-title">Ảnh gốc - Kích thước đầy đủ</div>', unsafe_allow_html=True)
                st.image(image, caption="Ảnh y tế gốc", use_container_width=True, output_format="PNG")
                if st.button("Đóng ảnh kích thước đầy đủ", key="close_original"):
                    st.session_state.show_fullsize_original = False
        
        # Xử lý ảnh với mô hình
        with st.spinner("Đang xử lý ảnh..."):
            st.markdown('<div style="display:flex;justify-content:center;margin:20px 0;">', unsafe_allow_html=True)
            st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            output_image, input_resolution, output_resolution = process_image(model, image, device)
        
        # Hiển thị ảnh sau khi xử lý - phiên bản thu nhỏ (preview)
        st.markdown('<div class="sub-title">Ảnh sau khi làm nét</div>', unsafe_allow_html=True)
        
        preview_col1, preview_col2 = st.columns([1, 2])
        with preview_col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(output_image, caption="", width=250, output_format="PNG", clamp=True)
            if st.button("Xem ảnh nét kích thước đầy đủ", key="view_enhanced", help="Nhấp để xem ảnh sau khi làm nét ở kích thước đầy đủ"):
                st.session_state.show_fullsize_enhanced = True
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Modal cho ảnh đã làm nét kích thước đầy đủ
        if st.session_state.show_fullsize_enhanced:
            with st.container():
                st.markdown('<div class="sub-title">Ảnh đã làm nét - Kích thước đầy đủ</div>', unsafe_allow_html=True)
                st.image(output_image, caption="Ảnh y tế sau khi làm nét", use_container_width=True, output_format="PNG")
                if st.button("Đóng ảnh kích thước đầy đủ", key="close_enhanced"):
                    st.session_state.show_fullsize_enhanced = False
        
        # Hiển thị thông số về độ phân giải trong các ô metric đẹp hơn
        st.markdown('<div class="sub-title">Thông số chi tiết</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Độ phân giải ảnh gốc</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{input_resolution} điểm ảnh</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Kích thước ảnh gốc</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{image.width}x{image.height}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Độ phân giải ảnh sau khi làm nét</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{output_resolution} điểm ảnh</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Kích thước ảnh đã làm nét</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{output_image.width}x{output_image.height}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Tỷ lệ tăng</div>', unsafe_allow_html=True)
            increase_ratio = output_resolution / input_resolution
            st.markdown(f'<div class="metric-value">{increase_ratio:.2f}x</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Thiết bị xử lý</div>', unsafe_allow_html=True)
            device_name = "GPU" if torch.cuda.is_available() else "CPU"
            if torch.cuda.is_available():
                device_name += f" ({torch.cuda.get_device_name(0)})"
            st.markdown(f'<div class="metric-value">{device_name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tải xuống ảnh đã làm nét
        st.markdown('<div class="sub-title">Tải xuống ảnh đã làm nét</div>', unsafe_allow_html=True)
        
        # Chuyển ảnh thành bytes để tải xuống
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # Nút tải xuống
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            download_button = st.download_button(
                label="Tải xuống ảnh đã làm nét",
                data=byte_im,
                file_name="enhanced_medical_image.png",
                mime="image/png",
                key="download_button"
            )

with tab2:
    st.markdown('<div class="sub-title">Hướng dẫn sử dụng</div>', unsafe_allow_html=True)
    
    # Chia thành nhiều phần hướng dẫn với giao diện đẹp hơn
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown('<h3>Tải lên ảnh y tế</h3>', unsafe_allow_html=True)
        st.write("""
        - Nhấn vào nút "Browse files" hoặc kéo và thả ảnh vào khu vực tải lên
        - Ứng dụng hỗ trợ các định dạng ảnh: JPG, JPEG, PNG và BMP
        - Khuyến nghị sử dụng ảnh có kích thước vừa phải để đảm bảo tốc độ xử lý
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown('<h3>Xem và so sánh ảnh</h3>', unsafe_allow_html=True)
        st.write("""
        - Ảnh gốc và ảnh đã được làm nét sẽ hiển thị với kích thước thu nhỏ
        - Nhấn vào nút "Xem kích thước đầy đủ" để xem ảnh ở kích thước lớn hơn
        - Dễ dàng so sánh chất lượng giữa ảnh gốc và ảnh sau khi xử lý
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown('<h3>Xử lý ảnh</h3>', unsafe_allow_html=True)
        st.write("""
        - Ứng dụng tự động xử lý ảnh bằng mô hình làm nét ảnh y tế
        - Quá trình xử lý mất vài giây tùy thuộc vào kích thước ảnh
        - Mô hình sẽ tăng độ phân giải ảnh lên 4 lần so với ảnh gốc
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown('<h3>Lưu kết quả</h3>', unsafe_allow_html=True)
        st.write("""
        - Sau khi xử lý xong, bạn có thể tải xuống ảnh đã được làm nét
        - Nhấn vào nút "Tải xuống ảnh đã làm nét" để lưu ảnh về máy
        - Ảnh được lưu dưới định dạng PNG để đảm bảo chất lượng tốt nhất
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Thêm phần thông tin ứng dụng
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown('<h3>Lưu ý quan trọng</h3>', unsafe_allow_html=True)
    st.write("""
    - Ứng dụng hoạt động tốt nhất với các ảnh y tế như X-quang, CT scan, MRI, siêu âm, v.v.
    - Kích thước ảnh đầu vào không nên quá lớn để tránh tràn bộ nhớ, đặc biệt là khi chạy trên CPU
    - Hệ thống sử dụng mô hình làm nét thông minh được đào tạo đặc biệt cho ảnh y tế, giúp:
      * Bảo toàn các chi tiết quan trọng trong chẩn đoán
      * Làm rõ các cạnh và cấu trúc trong ảnh
      * Giảm nhiễu và các nhiễu động không mong muốn
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer đẹp hơn
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('© 2025 Hệ thống làm nét ảnh y tế | Được phát triển bằng công nghệ trí tuệ nhân tạo và Streamlit', unsafe_allow_html=True)
st.markdown('<div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">', unsafe_allow_html=True)
st.markdown('<span style="color: #3498DB;">Hỗ trợ kỹ thuật</span> • <span style="color: #3498DB;">Liên hệ</span> • <span style="color: #3498DB;">Chính sách bảo mật</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)