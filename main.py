import streamlit as st
import os
from app import register_authorized_speaker, is_authorized_speaker

# Constants
AUTHORIZED_USER_FOLDER = "authenticated_user"  # Path to the authorized speaker folder

# Streamlit app title
st.title("Speaker Recognition App")

# Register the authorized speaker (this can be done once, or dynamically)
authorized_embedding_avg = None
if os.path.exists(AUTHORIZED_USER_FOLDER):
    authorized_embedding_avg = register_authorized_speaker(AUTHORIZED_USER_FOLDER)
    st.success("Authorized speaker registered!")

# File uploader for the new audio sample
uploaded_file = st.file_uploader("Upload an audio file to test", type=["wav", "mp3"])

# Check if authorized_embedding_avg exists
if authorized_embedding_avg is None:
    st.error("No authorized speaker registered. Please add samples in the 'authenticated_user' folder.")
else:
    # Handle the uploaded file
    if uploaded_file is not None:
        # Save the uploaded file to a temporary path
        new_audio_path = f"temp_audio/{uploaded_file.name}"
        with open(new_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Check if the uploaded file matches the authorized speaker
        is_authorized, similarity = is_authorized_speaker(new_audio_path, authorized_embedding_avg)
        
        # Display the results
        if is_authorized:
            st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
        else:
            st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")
        
        # Clean up the temporary file
        os.remove(new_audio_path)
