import streamlit as st
import pandas as pd
import numpy as np
import os

# Cấu hình giao diện ứng dụng rộng toàn màn hình
st.set_page_config(page_title="Phần Mềm Quay Số Gauss & Tần Suất", layout="wide")
st.title("📊 Phần Mềm Quay Số Ngẫu Nhiên Theo Tần Suất & Gauss")

# File lưu trữ dữ liệu cục bộ trên máy Mac
DATA_FILE = "data_goc.csv"

# Khởi tạo file dữ liệu nếu chưa có
if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame(columns=[f"Num{i}" for i in range(1, 7)])
    df_init.to_csv(DATA_FILE, index=False)

def load_data():
    return pd.read_csv(DATA_FILE)

df_tho = load_data()

# CHIA GIAO DIỆN THÀNH 2 CỘT: Cột trái (Xử lý số) - Cột phải (Thống kê tần suất)
col_trai, col_phai = st.columns(2, gap="large")

with col_trai:
    st.header("📥 Nhập bộ số mới xuất hiện")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: n1 = st.number_input("Số 1", min_value=1, max_value=99, value=1, key="n1")
    with c2: n2 = st.number_input("Số 2", min_value=1, max_value=99, value=1, key="n2")
    with c3: n3 = st.number_input("Số 3", min_value=1, max_value=99, value=1, key="n3")
    with c4: n4 = st.number_input("Số 4", min_value=1, max_value=99, value=1, key="n4")
    with c5: n5 = st.number_input("Số 5", min_value=1, max_value=99, value=1, key="n5")
    with c6: n6 = st.number_input("Số 6", min_value=1, max_value=99, value=1, key="n6")

    if st.button("LƯU SỐ MỚI & QUAY SỐ TIẾP THEO", type="primary"):
        new_row = pd.DataFrame([[n1, n2, n3, n4, n5, n6]], columns=df_tho.columns)
        df_tho = pd.concat([df_tho, new_row], ignore_index=True)
        df_tho.to_csv(DATA_FILE, index=False)
        st.success("🎉 Đã lưu bộ số mới và tự động cập nhật tần suất!")
        st.rerun()

# XỬ LÝ DỮ LIỆU THỐNG KÊ VÀ QUAY SỐ
if len(df_tho) > 0:
    # Gom tất cả các cột thành 1 mảng dọc, chuyển thành chuỗi số và loại bỏ ô trống một cách an toàn
    tat_ca_so = pd.to_numeric(df_tho.values.flatten(), errors='coerce')
    tat_ca_so = tat_ca_so[~np.isnan(tat_ca_so)].astype(int)
    
    # Tính tần suất chi tiết
    if len(tat_ca_so) > 0:
        tan_suat_series = pd.Series(tat_ca_so).value_counts()
        
        # Bảng xếp hạng từ thấp đến cao phục vụ bộ lọc Gauss
        tan_suat_tang_dan = tan_suat_series.sort_values(ascending=True)
        kho_so = tan_suat_tang_dan.index.tolist()
        
        # --- HIỂN THỊ CỘT TRÁI: KẾT QUẢ QUAY SỐ GAUSS ---
        with col_trai:
            st.write(f"Tổng số lượng con số độc nhất đang có: **{len(kho_so)} số**")
            if len(kho_so) >= 6:
                # THUẬT TOÁN GAUSS
                mean_idx = len(kho_so) / 2
                std_dev = len(kho_so) / 6
                
                so_da_chon = []
                vong_lap_bao_ve = 0
                
                while len(so_da_chon) < 6 and vong_lap_bao_ve < 1000:
                    vong_lap_bao_ve += 1
                    gauss_idx = int(np.round(np.random.normal(mean_idx, std_dev)))
                    if gauss_idx < 0: gauss_idx = 0
                    if gauss_idx >= len(kho_so): gauss_idx = len(kho_so) - 1
                    
                    so_chon = kho_so[gauss_idx]
                    if so_chon not in so_da_chon:
                        so_da_chon.append(so_chon)
                
                so_da_chon.sort()
                
                st.header("🔮 Kết quả dự đoán kỳ tiếp theo (Gauss):")
                cols_kq = st.columns(6)
                for idx, num in enumerate(so_da_chon):
                    with cols_kq[idx]:
                        st.markdown(f"<h1 style='text-align: center; color: #FFFFFF; background-color: #FF4B4B; padding: 10px; border-radius: 50%; width: 75px; height: 75px; line-height: 55px; margin: auto; font-size: 28px;'>{num:02d}</h1>", unsafe_allow_html=True)
            else:
                st.warning("Kho số hiện tại chưa đủ 6 số khác nhau để phân tích Gauss.")

        # --- HIỂN THỊ CỘT PHẢI: BẢNG THỐNG KÊ TOP XUẤT HIỆN ---
        with col_phai:
            st.header("📈 Bảng Thống Kê Tần Suất")
            
            # Chuẩn bị dữ liệu hiển thị trực quan
            df_thong_ke = pd.DataFrame({
                "Con Số": tan_suat_series.index,
                "Số Lần Xuất Hiện": tan_suat_series.values
            })
            
            sub_col1, sub_col2 = st.columns(2)
            
            with sub_col1:
                st.subheader("🔥 Top 10 số NHIỀU nhất")
                df_hot = df_thong_ke.sort_values(by=["Số Lần Xuất Hiện", "Con Số"], ascending=[False, True]).head(10)
                st.dataframe(df_hot, hide_index=True, use_container_width=True)
                
            with sub_col2:
                st.subheader("❄️ Top 10 số ÍT nhất")
                df_cold = df_thong_ke.sort_values(by=["Số Lần Xuất Hiện", "Con Số"], ascending=[True, True]).head(10)
                st.dataframe(df_cold, hide_index=True, use_container_width=True)
    else:
        with col_trai: st.info("Ứng dụng đang trống. Hãy nhập bộ số đầu tiên ở trên để bắt đầu.")
else:
    with col_trai: st.info("Ứng dụng đang trống. Hãy nhập bộ số đầu tiên ở trên để bắt đầu.")
