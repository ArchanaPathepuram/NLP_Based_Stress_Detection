import streamlit as st
import tensorflow as tf
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Download NLTK Resources
# ----------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

# ----------------------------
# Constants
# ----------------------------
MAX_LEN = 153

# ----------------------------
# Load Model
# ----------------------------
# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("stress_detection_bilstm.new.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# ----------------------------

# ----------------------------
# Load Tokenizer (Cached)
# ----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer (3).pkl", "rb") as file:
        return pickle.load(file)

try:
    tokenizer = load_tokenizer()
except Exception as e:
    st.error(f"Unable to load tokenizer: {e}")
    st.stop()
# ----------------------------
# NLP Tools
# ----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Text Preprocessing Function
# ----------------------------
def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    words = text.split()

    # Remove Stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# ----------------------------
# Prediction Function
# ----------------------------
def predict_stress(text):

    processed = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([processed])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="pre",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = "Stress"
    else:
        label = "No Stress"

    return label, probability, processed

# ----------------------------
# Streamlit UI
# ----------------------------

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("🧠 Stress Detection")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project Information")

st.sidebar.write("**Model:** Bidirectional LSTM")
st.sidebar.write("**Embedding:** GloVe (100 Dimensions)")
st.sidebar.write("**Dataset:** Stress.csv")
st.sidebar.write(f"**Maximum Sequence Length:** {MAX_LEN}")

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ This application is developed for educational purposes only and should not be considered a medical diagnosis tool."
)

st.sidebar.subheader("📊 Model Performance")

st.sidebar.metric(
    label="Accuracy",
    value="71.25%"
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ NLP Pipeline")

st.sidebar.markdown("""
- Lowercase Conversion
- URL Removal
- HTML Tag Removal
- Punctuation Removal
- Stopword Removal
- Lemmatization
- Tokenization
- Sequence Padding
- Bi-LSTM Prediction
""")

st.sidebar.markdown("---")



# ==========================================================
# Main Page
# ==========================================================

st.title("🧠 Stress Detection System")

st.subheader("AI-Powered Mental Stress Prediction using NLP & Bidirectional LSTM")

st.markdown("---")

st.info(
    "Enter any sentence below. The model predicts whether the text expresses **Stress** or **No Stress**."
)

# ----------------------------------------------------------
# Example Sentences
# ----------------------------------------------------------

st.subheader("💡 Try an Example")

col1, col2 = st.columns(2)

with col1:
    if st.button("😔 Stress Example"):
        st.session_state.text = (
            "For the past week I've barely been sleeping because I'm terrified I'll lose my job."
        )

with col2:
    if st.button("😊 No Stress Example"):
        st.session_state.text = (
            "I spent the afternoon hiking with friends and enjoyed the beautiful weather."
        )

# ----------------------------------------------------------
# Input Area
# ----------------------------------------------------------

if "text" not in st.session_state:
    st.session_state.text = ""

user_input = st.text_area(
    "Enter Text",
    value=st.session_state.text,
    height=180,
    placeholder="Example: I feel overwhelmed because of my work."
)

# ----------------------------------------------------------
# Buttons
# ----------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    predict = st.button("🔍 Predict")

with col2:
    clear = st.button("🗑 Clear")

if clear:
    st.session_state.text = ""
    st.rerun()

# ----------------------------------------------------------
# Input Statistics
# ----------------------------------------------------------

if user_input:

    c1, c2 = st.columns(2)

    c1.metric(
        "Words",
        len(user_input.split())
    )

    c2.metric(
        "Characters",
        len(user_input)
    )

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

if predict:

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        with st.spinner("Analyzing text..."):

            label, probability, processed = predict_stress(user_input)

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        if label == "Stress":

            st.error("⚠ Stress Detected")

        else:

            st.success("✅ No Stress Detected")

        c1, c2 = st.columns(2)

        c1.metric(
            "Stress Probability",
            f"{probability*100:.2f}%"
        )
        
        c2.metric(
            "Prediction",
            label
        )

        st.progress(probability)

        # --------------------------------------

        if probability >= 0.90:
            confidence = "Very High"

        elif probability >= 0.75:
            confidence = "High"

        elif probability >= 0.60:
            confidence = "Moderate"

        else:
            confidence = "Low"

        st.write(f"**Confidence Level:** {confidence}")

        # --------------------------------------

        st.subheader("📝 Processed Text")

        st.code(processed)

        # --------------------------------------

        st.subheader("📋 Prediction Summary")

        summary = {
            "Prediction": label,
            "Confidence": f"{probability*100:.2f}%",
            "Processed Words": len(processed.split()),
            "Input Words": len(user_input.split())
        }
        
        st.json(summary)
# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.markdown("---")

st.caption(
    "Developed using TensorFlow, GloVe Embeddings, Bidirectional LSTM and Streamlit."
)