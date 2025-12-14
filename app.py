import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="Sistem Cerdas Toko Bangunan", page_icon="🏗️", layout="wide")

# Styling CSS untuk tampilan yang lebih menarik
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #2c3e50; }
    [data-testid="stSidebar"] * { color: white !important; }
    .stButton>button { 
        background: linear-gradient(135deg, #e67e22, #d35400); 
        color: white; 
        width: 100%; 
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    h1 { color: #d35400; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 2. LOAD DATA & MODEL
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_rf_harga.pkl")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        # Bersihkan nama kolom agar konsisten dengan training (Spasi -> Underscore)
        df.columns = df.columns.str.replace(' ', '_')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    try:
        # Load model dan daftar fitur yang tersimpan
        model, features = joblib.load(MODEL_PATH)
        return model, features
    except FileNotFoundError:
        return None, None

df = load_data()
model, feature_columns = load_model()

# Cek apakah file ada
if df is None:
    st.error("❌ File CSV tidak ditemukan! Pastikan file 'TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv' ada di folder yang sama.")
    st.stop()

if model is None:
    st.error("❌ File Model PKL tidak ditemukan! Pastikan file 'model_rf_harga.pkl' sudah diupload.")
    st.stop()

# ==================================================
# 3. SIDEBAR MENU
# ==================================================
st.sidebar.title("🏗️ Navigasi")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Pilih Menu:", ["🏠 Home", "💰 Prediksi Harga", "📊 Segmentasi Pelanggan"])

# ==================================================
# 4. MENU: HOME
# ==================================================
if menu == "🏠 Home":
    st.title("🏗️ Sistem Cerdas Toko Bangunan")
    st.image("https://img.freepik.com/free-vector/construction-shop-building-materials-store-facade_107791-3254.jpg", use_column_width=True)
    st.markdown("""
    ### Selamat Datang!
    Aplikasi ini menggunakan **Machine Learning** untuk membantu operasional toko:
    1. **Prediksi Harga Total**: Menggunakan algoritma *Random Forest Regressor*.
    2. **Segmentasi Pelanggan**: Menggunakan algoritma *K-Means Clustering*.
    """)

# ==================================================
# 5. MENU: PREDIKSI HARGA (REGRESI)
# ==================================================
elif menu == "💰 Prediksi Harga":
    st.title("💰 Prediksi Total Harga")
    st.markdown("Masukkan detail pembelian untuk memprediksi total harga.")

    # Input User
    col1, col2 = st.columns(2)
    with col1:
        qty = st.number_input("📦 Jumlah Barang (Pcs)", min_value=1, value=10)
        harga = st.number_input("🏷️ Harga Satuan (Rp)", min_value=100, value=50000, step=1000)
    with col2:
        # Pilihan Kategori (Nama harus sama dengan yang ada di data CSV asli)
        opsi_kategori = ["Alat", "Bahan Logam dan PVC", "Cat", "Material Konstruksi"]
        kategori_pilihan = st.selectbox("📂 Kategori Produk", opsi_kategori)

    if st.button("HITUNG PREDIKSI 🚀"):
        try:
            # 1. Siapkan Template Input (Semua fitur diisi 0 dulu)
            input_data = {col: [0] for col in feature_columns}

            # 2. Masukkan Data Numerik
            # Kita cari nama kolom yang cocok di model (Harga_Satuan atau Harga Satuan)
            if 'Harga_Satuan' in feature_columns:
                input_data['Harga_Satuan'] = [harga]
            elif 'Harga Satuan' in feature_columns:
                input_data['Harga Satuan'] = [harga]
            
            input_data['Kuantitas'] = [qty]

            # 3. Masukkan Data Kategori (One-Hot Encoding Manual)
            # Kita coba cari format nama kolom kategori yang cocok di model
            
            # Kemungkinan 1: Format Underscore (Kategori_Bahan_Logam_dan_PVC)
            kategori_underscore = f"Kategori_{kategori_pilihan.replace(' ', '_')}"
            # Kemungkinan 2: Format Asli (Kategori_Bahan Logam dan PVC)
            kategori_asli = f"Kategori_{kategori_pilihan}"

            if kategori_underscore in feature_columns:
                input_data[kategori_underscore] = [1]
            elif kategori_asli in feature_columns:
                input_data[kategori_asli] = [1]
            else:
                st.warning(f"⚠️ Kategori '{kategori_pilihan}' tidak ditemukan spesifik di model, prediksi mungkin menggunakan nilai rata-rata.")

            # 4. Buat DataFrame & Prediksi
            input_df = pd.DataFrame(input_data)
            
            # Pastikan urutan kolom SAMA PERSIS dengan saat training
            input_df = input_df[feature_columns]

            prediksi = model.predict(input_df)[0]
            
            st.success(f"💵 Estimasi Total Harga: **Rp {prediksi:,.0f}**")
            st.balloons()
            
        except Exception as e:
            st.error("Terjadi kesalahan teknis.")
            st.code(f"Error Detail: {e}")

# ==================================================
# 6. MENU: SEGMENTASI PELANGGAN (CLUSTERING)
# ==================================================
elif menu == "📊 Segmentasi Pelanggan":
    st.title("👥 Segmentasi Pelanggan")
    st.markdown("Mengelompokkan pelanggan berdasarkan perilaku belanja.")

    k = st.slider("Jumlah Cluster (Kelompok)", min_value=2, max_value=5, value=3)

    if st.button("PROSES CLUSTERING"):
        # Grouping data berdasarkan ID Transaksi
        # Pastikan menggunakan nama kolom yang sudah bersih (Total_Harga, Kuantitas)
        df_cluster = df.groupby("ID_Transaksi").agg({
            "Total_Harga": "sum",
            "Kuantitas": "sum"
        }).reset_index()

        # Normalisasi Data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_cluster[["Total_Harga", "Kuantitas"]])

        # Modelling K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_cluster["Cluster"] = kmeans.fit_predict(scaled_data)

        # Visualisasi
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df_cluster, 
            x="Total_Harga", 
            y="Kuantitas", 
            hue="Cluster", 
            palette="viridis", 
            s=100, 
            ax=ax
        )
        plt.title(f"Hasil Segmentasi ({k} Cluster)")
        plt.xlabel("Total Belanja (Rp)")
        plt.ylabel("Jumlah Barang (Pcs)")
        
        st.pyplot(fig)
        
        # Tampilkan Data Hasil
        st.write("### Data Hasil Clustering")
        st.dataframe(df_cluster.head())
