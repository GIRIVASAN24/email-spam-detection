import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📧 Email Spam Detector")
st.write("Enter an email message below to check if it's spam or not.")

user_input = st.text_area("Your Email Message:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: {result}")
