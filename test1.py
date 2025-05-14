import streamlit as st
import pandas as pd

# Khởi tạo các giá trị mặc định cho biến
Nname = ""
Nage = 0
Nemail = ""

# Tạo giao diện upload file Excel
uploaded_file = st.file_uploader("Chọn file Excel", type=["xlsx"])

# Nếu có file được tải lên
if uploaded_file is not None:
    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(uploaded_file)

    # Kiểm tra xem file có dữ liệu hay không và điền vào các form
    if not df.empty:
        # Nếu có cột "Tên", "Tuổi" và "Email" trong Excel
        if 'Tên' in df.columns and 'Tuổi' in df.columns and 'Email' in df.columns:
            # Lấy dữ liệu từ dòng đầu tiên của file Excel và cập nhật vào các biến
            n=1 
            Nname = df['Tên'].iloc[n]  # Điền tên vào form
            Nage = int(df['Tuổi'].iloc[n])  # Điền tuổi vào form
            Nemail = df['Email'].iloc[n]  # Điền email vào form

# Khởi tạo các trường đầu vào cho form với giá trị mặc định
name = st.text_input('Tên', Nname)
age = st.number_input('Tuổi', min_value=0, value=Nage)
email = st.text_input('Email', Nemail)

# Nút Submit
submit_button = st.button(label='Submit')

if submit_button:
    st.write(f"Đã nhập thông tin: Tên: {name}, Tuổi: {age}, Email: {email}")
