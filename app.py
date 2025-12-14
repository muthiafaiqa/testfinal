import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# KONFIGURASI HALAMAN & CSS
# ==================================================
st.set_page_config(page_title="Sistem Cerdas Toko Bangunan", page_icon="🏗️", layout="wide")
st.markdown("""
<style>
/* CSS Styling */
[data-testid="stSidebar"] { background-color: #2c3e50; }
[data-testid="stSidebar"] * { color: white !important; }
h1 { color: #e67e22; font-weight: 800; text-align: center; }
.card { background: white; padding: 25px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.08); }
.stButton>button { background: linear-gradient(135deg, #e67e22, #d35400); color: white; font-weight: bold; width: 100%; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# ==================================================
# PATH AMAN (CLOUD SAFE)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Nama file CSV yang sudah dibersihkan (semua spasi diganti _)
DATA_PATH = os.path.join(BASE_DIR, "TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_rf_harga.pkl")

# ==================================================
# LOAD DATA & MODEL
# ==================================================
@st.cache_data
def load_data():
    # MEMBACA DATA DAN MEMBERSIHKAN SPASI DI NAMA KOLOM
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(' ', '_')
    return df

@st.cache_resource
def load_model():
    # MEMUAT MODEL DAN DAFTAR FITUR BERSIH
    model, features = joblib.load(MODEL_PATH)
    return model, features

df = load_data()
model, feature_columns = load_model()

if df is None or model is None:
    st.error("⚠️ Aplikasi berhenti karena file model/data tidak ditemukan.")
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🏗️ Navigasi")
menu = st.sidebar.radio("Menu", ["🏠 Home", "💰 Prediksi Harga", "📊 Segmentasi Pelanggan"])

# ==================================================
# HOME
# ==================================================
if menu == "🏠 Home":
    st.title("🏗️ Sistem Cerdas Toko Bangunan")
    st.markdown("<div class='card'>Aplikasi Data Mining berbasis Machine Learning.</div>", unsafe_allow_html=True)

# ==================================================
# REGRESI (PREDIKSI HARGA) - Menggunakan UNDERSCORE
# ==================================================
# ==================================================
# REGRESI (PREDIKSI HARGA) - FIX FINAL 100% AMAN
# ==================================================
elif menu == "💰 Prediksi Harga":
    st.title("💰 Prediksi Total Harga")

    col1, col2 = st.columns(2)
    with col1:
        qty = st.number_input("Jumlah Barang", min_value=1, value=5)
        # WAJIB: Pastikan nama kolom numerik yang dikirim 100% cocok dengan PKL
        harga = st.number_input("Harga Satuan (Rp)", min_value=1000, value=50000, step=1000)
    with col2:
        # PENTING: Pilihan kategori harus menggunakan format yang mudah dibaca (dengan spasi)
        kategori_opsi = ["Alat", "Bahan Logam dan PVC", "Cat", "Material Konstruksi"] 
        kategori_pilihan = st.selectbox("Kategori", kategori_opsi)

    if st.button("HITUNG"):
        
        # 1. Buat Template DataFrame (SEMUA FITUR DIISI 0)
        input_dict = {col: [0] for col in feature_columns}

        # 2. Masukkan Nilai Numerik
        try:
            # Cari nama kolom numerik yang benar-benar ada di PKL: Harga_Satuan atau Harga Satuan?
            if "Harga_Satuan" in feature_columns:
                 input_dict["Harga_Satuan"] = [harga]
            elif "Harga Satuan" in feature_columns: # Jaga-jaga jika PKL masih menyimpan spasi
                 input_dict["Harga Satuan"] = [harga]
            
            input_dict["Kuantitas"] = [qty]

            # 3. AKTIFKAN KOLOM KATEGORI (Bagian paling rawan error)
            
            # Coba format yang bersih: Kategori_Bahan_Logam_dan_PVC
            nama_bersih = f"Kategori_{kategori_pilihan.replace(' ', '_')}"
            
            # Coba format yang kotor (asli): Kategori_Bahan Logam dan PVC
            nama_kotor = f"Kategori_{kategori_pilihan}"
            
            kolom_aktif = None
            
            # Cari nama yang benar-benar ada di daftar fitur model (feature_columns)
            if nama_bersih in feature_columns:
                kolom_aktif = nama_bersih
            elif nama_kotor in feature_columns:
                kolom_aktif = nama_kotor
            
            # Aktifkan kolom yang ditemukan
            if kolom_aktif:
                input_dict[kolom_aktif] = [1]
            else:
                st.error(f"Error: Tidak ada kolom kategori yang cocok di model. Cek nama fitur di PKL.")
                st.stop() # Hentikan proses jika kategori tidak ditemukan


            # 4. Buat DataFrame Input DENGAN URUTAN YANG BENAR
            input_df = pd.DataFrame(input_dict)
            input_df = input_df[feature_columns] # WAJIB: Memastikan urutan kolomnya sama persis!

            # 5. Prediksi
            pred = model.predict(input_df)[0]
            st.success(f"💵 Estimasi Total: Rp {pred:,.0f} (Menggunakan Random Forest)")
            st.balloons()
            
        except Exception as e:
            st.error("Terjadi kesalahan teknis yang parah pada prediksi.")
            st.code(f"Error detail: {e}")

# ==================================================
# CLUSTERING (SEGMENTASI) - Menggunakan UNDERSCORE
# ==================================================
elif menu == "📊 Segmentasi Pelanggan":
    st.title("👥 Segmentasi Pelanggan")

    k = st.slider("Jumlah Cluster", 2, 5, 3)

    # Proses Clustering menggunakan nama kolom yang sudah bersih
    df_cluster = df.groupby("ID_Transaksi").agg({"Total_Harga": "sum", "Kuantitas": "sum"}).reset_index() 

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_cluster[["Total_Harga", "Kuantitas"]])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_cluster, x="Total_Harga", y="Kuantitas", hue="Cluster", palette="viridis", s=120, ax=ax)
    plt.xlabel("Total Belanja (Rp)")
    plt.ylabel("Kuantitas")
    st.pyplot(fig)

    st.dataframe(df_cluster.head(), use_container_width=True)



