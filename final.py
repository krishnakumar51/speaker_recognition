import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import os
from io import BytesIO
from app import register_authorized_speaker, is_authorized_speaker

# Constants
AUTHORIZED_USER_FOLDER = "authenticated_user"  # Path to the authorized speaker folder
TEMP_AUDIO_FOLDER = "temp_audio"  # Path to save uploaded and recorded audio files temporarily

# Create temp_audio directory if it doesn't exist
if not os.path.exists(TEMP_AUDIO_FOLDER):
    os.makedirs(TEMP_AUDIO_FOLDER)

# Register the authorized speaker (this can be done once, or dynamically)
authorized_embedding_avg = None
if os.path.exists(AUTHORIZED_USER_FOLDER):
    authorized_embedding_avg = register_authorized_speaker(AUTHORIZED_USER_FOLDER)
    st.success("Authorized speaker registered!")

def save_audio_file(audio_data, file_name):
    file_path = os.path.join(TEMP_AUDIO_FOLDER, file_name)
    with open(file_path, "wb") as f:
        f.write(audio_data)
    return file_path

def record_audio(duration=5, samplerate=44100):
    """Record audio for a given duration."""
    st.write("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished.")
    return audio_data, samplerate

def save_wav(filename, audio_data, samplerate):
    """Save numpy array as a .wav file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

def play_audio(file_path):
    """Play audio file in Streamlit."""
    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

# Handle file uploads or recording
st.header("Test the Speaker Recognition System")
option = st.selectbox("Choose an option", ("Upload an audio file", "Record your voice"))

if option == "Upload an audio file":
    uploaded_file = st.file_uploader("Upload an audio file to test", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary path
        new_audio_path = save_audio_file(uploaded_file.read(), uploaded_file.name)
        
        if authorized_embedding_avg is not None:
            is_authorized, similarity = is_authorized_speaker(new_audio_path, authorized_embedding_avg)
            
            if is_authorized:
                st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
            else:
                st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")
        
        # Playback
        play_audio(new_audio_path)
        
        # Clean up the temporary file
        os.remove(new_audio_path)

elif option == "Record your voice":
    if st.button("Start Recording"):
        audio_data, samplerate = record_audio(duration=5)
        temp_audio_path = os.path.join(TEMP_AUDIO_FOLDER, "recorded_audio.wav")
        save_wav(temp_audio_path, audio_data, samplerate)
        
        st.write("Recording complete!")
        st.audio(temp_audio_path, format='audio/wav')
        
        # Check if the recorded file matches the authorized speaker
        if authorized_embedding_avg is not None:
            is_authorized, similarity = is_authorized_speaker(temp_audio_path, authorized_embedding_avg)
            
            if is_authorized:
                st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
            else:
                st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")

        # Clean up the temporary file
        os.remove(temp_audio_path)
