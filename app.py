import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter
# ==========================================
# CẤU HÌNH GITHUB API (Đọc cấu hình bảo mật)
# ==========================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]  # Dòng này GIỮ NGUYÊN (Streamlit tự nạp từ mục Secrets)

REPO_NAME = "nguyenminhtam19972007-spec/xosovietlot"  # 🔴 BẮT BUỘC SỬA: Thay bằng Tên GitHub/Tên Repo của bạn
FILE_PATH = "data_goc.csv"  # 🔴 SỬA NẾU CẦN: Đường dẫn nếu file nằm trong thư mục con (VD: "data/data_goc.csv")
BRANCH_NAME = "main"  # 🔴 SỬA NẾU CẦN: Đổi thành "master" nếu repo của bạn dùng tên nhánh cũ

# 1. Cấu hình giao diện ứng dụng trên iPhone
st.set_page_config(page_title="Phần Mềm Gauss Ma Trận Cặp Số", layout="wide")
st.title("🔮 Phần Mềm Gauss Ma Trận Tím (Dành Cho iPhone & Máy Tính)")

# Danh sách số trong hình gốc của bạn để nạp làm nền tảng ban đầu
SO_TRONG_HINH = [
    6, 12, 24, 30, 38, 40,
    1, 10, 21, 27, 34, 43,
    3, 15, 19, 26, 32, 36,
    2, 11, 16, 25, 31, 45,
    4, 8, 14, 22, 35, 42,
    7, 13, 20, 29,41, 44
]

DATA_FILE = "data_goc.csv"

# Tự động nạp sẵn dữ liệu từ ảnh gốc nếu chưa có file
if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame([,
 ,
 ,
 ,
 ,
        [7, 13, 20, 29, 38, 44]
    ], columns=[f"Num{i}" for i in range(1, 7)])
    df_init.to_csv(DATA_FILE, index=False)

def load_data():
    return pd.read_csv(DATA_FILE)

df_tho = load_data()

# Khởi tạo bộ nhớ tạm để khóa kết quả dự đoán trên iPhone
if "ket_qua_du_doan" not in st.session_state:
    st.session_state.ket_qua_du_doan = None

# CHIA GIAO DIỆN THÀNH 2 CỘT (Tự động xếp dọc khi xem trên màn hình đứng của iPhone)
col_trai, col_phai = st.columns(2, gap="large")

with col_trai:
    st.header("📥 Nhập bộ số mới xuất hiện")
    
    # BIỂU MẪU KHÓA (Form): Nhập xong xuôi bấm nút mới tính toán
    with st.form("my_input_form", clear_on_submit=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: n1 = st.number_input("Số 1", min_value=1, max_value=55, value=6, key="n1")
        with c2: n2 = st.number_input("Số 2", min_value=1, max_value=55, value=12, key="n2")
        with c3: n3 = st.number_input("Số 3", min_value=1, max_value=55, value=24, key="n3")
        with c4: n4 = st.number_input("Số 4", min_value=1, max_value=55, value=30, key="n4")
        with c5: n5 = st.number_input("Số 5", min_value=1, max_value=55, value=38, key="n5")
        with c6: n6 = st.number_input("Số 6", min_value=1, max_value=55, value=40, key="n6")
        
        submitted = st.form_submit_button("LƯU SỐ MỚI & GOM DÃY SỐ MA TRẬN", type="primary")

    if submitted:
        bo_so_moi = [n1, n2, n3, n4, n5, n6]
        if len(set(bo_so_moi)) < 6:
            st.error("❌ Vui lòng không nhập trùng các con số trong cùng một kỳ!")
        else:
            # 1. Ghi dữ liệu mới vào file lưu trữ
            new_row = pd.DataFrame([[n1, n2, n3, n4, n5, n6]], columns=df_tho.columns)
            df_tho = pd.concat([df_tho, new_row], ignore_index=True)
            df_tho.to_csv(DATA_FILE, index=False)
            st.success("🎉 Đã lưu bộ số mới thành công!")
            
            # 2. Thuật toán phân tích Ma trận cặp số kết hợp chuông Gauss
            cap_so_counts = Counter()
            for idx, row in df_tho.iterrows():
                hang_so = row.dropna().astype(int).tolist()
                for i in range(len(hang_so)):
                    for j in range(i + 1, len(hang_so)):
                        s1, s2 = min(hang_so[i], hang_so[j]), max(hang_so[i], hang_so[j])
                        cap_so_counts[(s1, s2)] += 1
            
            tat_ca_so = pd.to_numeric(df_tho.values.flatten(), errors='coerce')
            tat_ca_so = tat_ca_so[~np.isnan(tat_ca_so)].astype(int)
            tan_suat_series = pd.Series(tat_ca_so).value_counts().sort_values(ascending=True)
            kho_so = tan_suat_series.index.tolist()
            
            if len(kho_so) >= 6:
                bo_so_hop_le = False
                so_da_chon = []
                vong_lap_toi_uu = 0
                
                # Vòng lặp kén chọn bộ số đỉnh chuông Gauss
                while not bo_so_hop_le and vong_lap_toi_uu < 5000:
                    vong_lap_toi_uu += 1
                    so_da_chon = []
                    
                    # Bước A: Chọn 1 số làm Hồng tâm bằng phân phối Gauss
                    mean_idx = len(kho_so) / 2
                    std_dev = len(kho_so) / 6
                    gauss_idx = int(np.round(np.random.normal(mean_idx, std_dev)))
                    if gauss_idx < 0: gauss_idx = 0
                    if gauss_idx >= len(kho_so): gauss_idx = len(kho_so) - 1
                    goc_so = kho_so[gauss_idx]
                    so_da_chon.append(goc_so)
                    
                    # Bước B: Kéo các cặp bài trùng bám dính nhất vào chung dãy số
                    diem_than_thiet = {}
                    for so_khac in kho_so:
                        if so_khac != goc_so:
                            pair = (min(goc_so, so_khac), max(goc_so, so_khac))
                            diem_than_thiet[so_khac] = cap_so_counts.get(pair, 0)
                    
                    cac_so_than_thiet = sorted(diem_than_thiet.items(), key=lambda x: x[1], reverse=True)
                    
                    idx_than = 0
                    while len(so_da_chon) < 6 and idx_than < len(cac_so_than_thiet):
                        ung_vien = cac_so_than_thiet[idx_than][0]
                        if ung_vien not in so_da_chon:
                            so_da_chon.append(ung_vien)
                        idx_than += 1
                    
                    # Bước C: Kiểm tra qua màng lọc Tổng vàng (110-210) và tỷ lệ Chẵn/Lẻ cân bằng
                    if len(so_da_chon) == 6:
                        tong_so = sum(so_da_chon)
                        if 110 <= tong_so <= 210:
                            so_chan = sum(1 for x in so_da_chon if x % 2 == 0)
                            if so_chan in:
                                bo_so_hop_le = True
                
                so_da_chon.sort()
                st.session_state.ket_qua_du_doan = so_da_chon
                st.rerun()

# HIỂN THỊ DÃY SỐ MÀU TÍM SAU KHI ĐÃ ĐƯỢC KHÓA CỐ ĐỊNH
with col_trai:
    if st.session_state.ket_qua_du_doan is not None:
        st.header("🔮 Dãy số dự đoán đã gom cụm Ma trận:")
        cols_kq = st.columns(6)
        for idx, num in enumerate(st.session_state.ket_qua_du_doan):
            with cols_kq[idx]:
                st.markdown(f"<h1 style='text-align: center; color: #FFFFFF; background-color: #7928CA; padding: 10px; border-radius: 50%; width: 75px; height: 75px; line-height: 55px; margin: auto; font-size: 28px;'>{num:02d}</h1>", unsafe_allow_html=True)
        st.caption("🔮 Bóng số màu TÍM QUÝ TỘC: Biểu thị dãy số được tính toán dựa trên các cặp số hay về chung một kỳ nhiều nhất.")
    else:
        st.info("Hãy điền dữ liệu kỳ mới nhất vào biểu mẫu phía trên và bấm nút để bắt đầu phân tích.")

# --- HIỂN THỊ CỘT PHẢI: BẢNG THỐNG KÊ CẶP BÀI TRÙNG BÁM DÍNH ---
with col_phai:
    st.header("📈 Bảng Thống Kê Cặp Bài Trùng")
    if len(df_tho) > 0:
        cap_counts = Counter()
        for idx, row in df_tho.iterrows():
            hang_so = row.dropna().astype(int).tolist()
            for i in range(len(hang_so)):
                for j in range(i + 1, len(hang_so)):
                    s1, s2 = min(hang_so[i], hang_so[j]), max(hang_so[i], hang_so[j])
                    cap_counts[f"{s1:02d} - {s2:02d}"] += 1
        
        if len(cap_counts) > 0:
            df_pairs = pd.DataFrame({
                "Cặp Số Hay Đi Chung": list(cap_counts.keys()),
                "Số Kỳ Xuất Hiện Cùng Nhau": list(cap_counts.values())
            }).sort_values(by="Số Kỳ Xuất Hiện Cùng Nhau", ascending=False)
            
            st.subheader("🔥 Top các cặp bài trùng bám dính nhau nhất")
            st.dataframe(df_pairs.head(15), hide_index=True, use_container_width=True)
