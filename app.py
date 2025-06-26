import streamlit as st
import joblib
import pandas as pd  # ✅ Tambahan
from preprocessing import preprocess_input

# Load model
model = joblib.load("model/knn_model.pkl")

st.title("Prediksi Risiko Diabetes (CDC Dataset)")

st.markdown("Masukkan data diri Anda untuk mengetahui risiko diabetes.")

# Form input
with st.form("form_diabetes"):
    col1, col2 = st.columns(2)

    with col1:
        HighBP = st.selectbox("Tekanan Darah Tinggi", [0, 1])
        HighChol = st.selectbox("Kolesterol Tinggi", [0, 1])
        CholCheck = st.selectbox("Tes Kolesterol", [0, 1])
        Smoker = st.selectbox("Perokok", [0, 1])
        Stroke = st.selectbox("Pernah Stroke", [0, 1])
        HeartDiseaseorAttack = st.selectbox("Penyakit Jantung/Serangan Jantung", [0, 1])
        PhysActivity = st.selectbox("Aktivitas Fisik", [0, 1])
    
    with col2:
        Fruits = st.selectbox("Konsumsi Buah", [0, 1])
        Veggies = st.selectbox("Konsumsi Sayur", [0, 1])
        HvyAlcoholConsump = st.selectbox("Konsumsi Alkohol Berat", [0, 1])
        AnyHealthcare = st.selectbox("Punya Akses Kesehatan", [0, 1])
        NoDocbcCost = st.selectbox("Tidak ke dokter karena biaya", [0, 1])
        DiffWalk = st.selectbox("Kesulitan Berjalan", [0, 1])
        Sex = st.selectbox("Jenis Kelamin (0=Perempuan, 1=Laki-laki)", [0, 1])

    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    MentHlth = st.slider("Hari Tidak Sehat Mental (0-30)", 0, 30)
    PhysHlth = st.slider("Hari Tidak Sehat Fisik (0-30)", 0, 30)
    GenHlth = st.slider("Kesehatan Umum (1=Baik → 5=Buruk)", 1, 5)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_dict = {
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": CholCheck,
        "Smoker": Smoker,
        "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost,
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "BMI": BMI,
        "MentHlth": MentHlth,
        "PhysHlth": PhysHlth,
        "GenHlth": GenHlth
    }

    vector = preprocess_input(input_dict)

    input_df = pd.DataFrame([vector], columns=[
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
        "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "DiffWalk", "Sex", "BMI", "MentHlth", "PhysHlth", "GenHlth"
    ])

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("⚠️ Anda terdeteksi memiliki risiko diabetes.")
    else:
        st.success("✅ Anda TIDAK terdeteksi memiliki risiko diabetes.")
