import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Sistem Cerdas Toko Bangunan",
    page_icon="🏗️",
    layout="wide"
)

# ==================================================
# CSS CUSTOM
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f6f9, #ffffff);
}

[data-testid="stSidebar"] {
    background-color: #2c3e50;
}

[data-testid="stSidebar"] * {
    color: #ecf0f1 !important;
}

h1 {
    color: #e67e22;
    text-align: center;
    font-weight: 800;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton>button {
    background: linear-gradient(135deg, #e67e22, #d35400);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 28px;
    width: 100%;
    border: none;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.25);
}

.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA & MODEL
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")

@st.cache_resource
def load_model():
    model, feature_columns = joblib.load("model_rf_harga.pkl")
    return model, feature_columns

df = load_data()
model, feature_columns = load_model()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🏗️ Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["🏠 Home", "💰 Prediksi Harga", "📊 Segmentasi Pelanggan"]
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 | Data Mining Project")

# ==================================================
# HOME
# ==================================================
if menu == "🏠 Home":
    st.title("🏗️ Sistem Cerdas Toko Bangunan")

    st.markdown("""
    <div class="card">
        <h3>✨ Deskripsi Aplikasi</h3>
        <p>
        Aplikasi ini menggunakan <b>Machine Learning</b> untuk membantu
        pengambilan keputusan pada toko bangunan.
        </p>
        <ul>
            <li>🤖 <b>Regresi (Supervised)</b> – Prediksi total harga belanja</li>
            <li>👥 <b>Clustering (Unsupervised)</b> – Segmentasi pelanggan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# PREDIKSI HARGA (REGRESI)
# ==================================================
elif menu == "💰 Prediksi Harga":
    st.title("💰 Kasir Cerdas – Prediksi Harga")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        qty = st.number_input("📦 Jumlah Barang", min_value=1, value=5)
        harga_satuan = st.number_input(
            "💵 Harga Satuan (Rp)", min_value=1000, value=50000, step=1000
        )

    with col2:
        kategori = st.selectbox(
            "🏷️ Kategori Barang",
            ["Alat", "Bahan Logam dan PVC", "Cat", "Material Konstruksi"]
        )

    st.write("")

    if st.button("🔍 HITUNG ESTIMASI HARGA"):
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_data['Harga Satuan'] = harga_satuan
        input_data['Kuantitas'] = qty

        if kategori == "Alat":
            input_data['Kategori_Alat'] = 1
        elif kategori == "Bahan Logam dan PVC":
            input_data['Kategori_Bahan Logam dan PVC'] = 1
        elif kategori == "Cat":
            input_data['Kategori_Cat'] = 1
        elif kategori == "Material Konstruksi":
            input_data['Kategori_Material Konstruksi'] = 1

        prediction = model.predict(input_data)[0]
        st.success(f"🏷️ Total Estimasi Belanja: **Rp {prediction:,.0f}**")
        st.balloons()

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# CLUSTERING PELANGGAN (UNSUPERVISED)
# ==================================================
elif menu == "📊 Segmentasi Pelanggan":
    st.title("👥 Segmentasi Pelanggan")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    n_cluster = st.slider("🔢 Jumlah Cluster", 2, 5, 3)

    # Agregasi data per transaksi
    df_cluster = df.groupby("ID Transaksi").agg({
        "Total Harga": "sum",
        "Kuantitas": "sum"
    }).reset_index()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(
        df_cluster[['Total Harga', 'Kuantitas']]
    )

    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(scaled_data)

    st.info(f"Pelanggan dikelompokkan menjadi **{n_cluster} segmen**")

    # Visualisasi
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_cluster,
        x="Total Harga",
        y="Kuantitas",
        hue="Cluster",
        palette="viridis",
        s=120,
        ax=ax
    )
    ax.set_title("Peta Segmentasi Pelanggan")
    ax.set_xlabel("Total Belanja (Rp)")
    ax.set_ylabel("Jumlah Barang")
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)

    st.write("### 📝 Contoh Data Hasil Cluster")
    st.dataframe(df_cluster.head(), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
