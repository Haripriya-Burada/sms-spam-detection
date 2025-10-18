import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ---------------------- Text Preprocessing ----------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ---------------------- Load Model ----------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------------- Streamlit App UI ----------------------
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✉️ Enter the message")

if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before prediction.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display with simple emojis
        if result == 1:
            st.markdown("🚨 **Spam Message Detected!**")
        else:
            st.markdown("✅ **Not Spam — Safe Message!**")
