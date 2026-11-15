import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Analisis Sentimen Karhutla", page_icon="🔥", layout="wide")

BASE = Path(__file__).parent
HASIL = BASE / "hasil"
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}


# ---------- Fungsi bantu ----------
def bersihkan_teks(teks):
    """Pembersihan teks, sama persis dengan yang dipakai di Colab."""
    teks = str(teks)
    teks = re.sub(r":[a-z0-9_\-]+:", " ", teks, flags=re.I)  # kode emoji YouTube
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"@[\w\-\.]+", " ", teks)
    teks = re.sub(r"#\w+", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = teks.lower()
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def tampilkan_gambar(nama_file, keterangan=None):
    path = HASIL / nama_file
    if path.exists():
        st.image(str(path), caption=keterangan, use_container_width=True)
    else:
        st.warning(f"File gambar tidak ditemukan: hasil/{nama_file}")


@st.cache_data
def muat_data():
    df = pd.read_csv(BASE / "tweet_data_labeled.csv")
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    return df


@st.cache_resource
def latih_model(df):
    """Latih ulang model dengan pengaturan yang sama seperti di Colab (80:20, TF-IDF, MLP)."""
    X_train, _, y_train, _ = train_test_split(
        df["clean_text"], df["sentimen"], test_size=0.2, random_state=42, stratify=df["sentimen"]
    )
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_train_vec = tfidf.fit_transform(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=(100,), activation="relu", solver="adam",
        max_iter=300, random_state=42,
    )
    model.fit(X_train_vec, y_train)
    return model, tfidf


# ---------- Muat data dan model ----------
try:
    df = muat_data()
except FileNotFoundError:
    st.error("File tweet_data_labeled.csv tidak ditemukan. Letakkan di folder yang sama dengan app_sentimen_nn.py.")
    st.stop()

model, tfidf = latih_model(df)

# ---------- Tampilan ----------
st.title("🔥 Analisis Sentimen Komentar YouTube: Topik Kebakaran Hutan (Karhutla)")
st.caption("Metode: pelabelan rule-based, TF-IDF, dan Neural Network (Multilayer Perceptron)")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data & Distribusi", "☁️ WordCloud", "📈 Evaluasi Model", "🔮 Prediksi Teks Baru"]
)

# ----- Tab 1: Data dan distribusi -----
with tab1:
    jumlah = df["sentimen"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total komentar", len(df))
    c2.metric("😊 Positif", int(jumlah.get("positif", 0)))
    c3.metric("😐 Netral", int(jumlah.get("netral", 0)))
    c4.metric("😠 Negatif", int(jumlah.get("negatif", 0)))

    kiri, kanan = st.columns(2)
    with kiri:
        tampilkan_gambar("distribusi_bar.png", "Diagram batang distribusi sentimen")
    with kanan:
        tampilkan_gambar("distribusi_pie.png", "Diagram lingkaran distribusi sentimen")

    st.subheader("Contoh data")
    pilihan = st.selectbox("Filter sentimen", ["semua", "positif", "netral", "negatif"])
    tampil = df if pilihan == "semua" else df[df["sentimen"] == pilihan]
    st.dataframe(tampil[["full_text", "clean_text", "sentimen"]].head(100), use_container_width=True)

# ----- Tab 2: WordCloud -----
with tab2:
    st.subheader("Kata yang paling sering muncul di tiap kelas sentimen")
    k1, k2, k3 = st.columns(3)
    with k1:
        tampilkan_gambar("wordcloud_positif.png", "Positif")
    with k2:
        tampilkan_gambar("wordcloud_netral.png", "Netral")
    with k3:
        tampilkan_gambar("wordcloud_negatif.png", "Negatif")

# ----- Tab 3: Evaluasi -----
with tab3:
    st.subheader("Metrik performa model")
    m1, m2 = st.columns(2)
    with m1:
        f_total = HASIL / "metrik_total.csv"
        if f_total.exists():
            st.markdown("**Metrik keseluruhan**")
            st.dataframe(pd.read_csv(f_total), use_container_width=True, hide_index=True)
    with m2:
        f_kelas = HASIL / "metrik_per_kelas.csv"
        if f_kelas.exists():
            st.markdown("**Metrik per kelas**")
            st.dataframe(pd.read_csv(f_kelas, index_col=0), use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        tampilkan_gambar("confusion_matrix.png", "Confusion matrix")
    with g2:
        tampilkan_gambar("metrik_per_kelas.png", "Presisi, recall, dan F1-score per kelas")

# ----- Tab 4: Prediksi -----
with tab4:
    st.subheader("Coba prediksi sentimen teks baru")
    teks_input = st.text_area(
        "Masukkan komentar atau kalimat:", height=120,
        placeholder="Contoh: semoga pemerintah berani menindak pembakar hutan",
    )

    if st.button("Prediksi sentimen", type="primary"):
        bersih = bersihkan_teks(teks_input)
        if not bersih:
            st.warning("Teks kosong setelah dibersihkan. Coba masukkan kalimat yang berisi kata.")
        else:
            vec = tfidf.transform([bersih])
            proba = model.predict_proba(vec)[0]
            kelas = model.classes_
            label = kelas[proba.argmax()]

            st.markdown(f"### {EMOJI[label]} Sentimen: **{label.upper()}**")
            st.metric("Keyakinan model", f"{proba.max() * 100:.1f}%")

            tabel = pd.DataFrame(
                {"Sentimen": [f"{EMOJI[k]} {k}" for k in kelas], "Probabilitas": proba}
            ).set_index("Sentimen")
            st.bar_chart(tabel)
            st.dataframe(tabel.style.format("{:.1%}"), use_container_width=True)

            if vec.nnz == 0:
                st.info("Tidak ada kata dalam kalimat ini yang dikenal model, jadi hasilnya kurang bisa diandalkan.")
            with st.expander("Teks setelah dibersihkan"):
                st.write(bersih)
