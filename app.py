import streamlit as st
import whisper
import tempfile
import os

# Load the Whisper model
model = whisper.load_model("base")

# Set the title of the Streamlit app
st.title("Whisper AI Transcription App")

# Allow the user to upload an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Perform the transcription using Whisper
    result = model.transcribe(temp_file_path)
    transcription = result["text"]

    # Display the transcription
    st.subheader("Transcription")
    st.write(transcription)

    # Clean up the temporary file
    os.remove(temp_file_path)
