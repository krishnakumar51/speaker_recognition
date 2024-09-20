import streamlit as st
import numpy as np
import wave
import os
import pyaudio
import tempfile
import matplotlib.pyplot as plt
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

def record_audio(duration=5, sample_rate=44100):
    """Record audio for a given duration using PyAudio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    st.write("Recording...")
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))

    st.write("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames)

def save_wav(filename, audio_data, sample_rate=44100):
    """Save numpy array as a .wav file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def plot_waveform(audio_data):
    """Plot the waveform of the audio data."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(audio_data)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

def process_audio(audio_path):
    if authorized_embedding_avg is not None:
        is_authorized, similarity = is_authorized_speaker(audio_path, authorized_embedding_avg)
        
        if is_authorized:
            st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
        else:
            st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")

# Handle file uploads or recording
st.header("Test the Speaker Recognition System")
option = st.radio("Choose an option", ("Upload an audio file", "Record your voice"))

if option == "Upload an audio file":
    uploaded_file = st.file_uploader("Upload an audio file to test", type=["wav", "mp3"])
    if uploaded_file:
        new_audio_path = save_audio_file(uploaded_file.read(), uploaded_file.name)
        st.audio(new_audio_path)
        
        # Read and plot the waveform
        with wave.open(new_audio_path, 'rb') as wf:
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        plot_waveform(audio_data)
        
        if st.button("Test Speaker Recognition"):
            process_audio(new_audio_path)
        
        os.remove(new_audio_path)

elif option == "Record your voice":
    if st.button("Start Recording"):
        audio_data = record_audio(duration=5)
        
        # Save the recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_AUDIO_FOLDER) as temp_file:
            save_wav(temp_file.name, audio_data)
            temp_audio_path = temp_file.name
        
        st.write("Recording complete!")
        st.audio(temp_audio_path)
        
        # Plot the waveform
        plot_waveform(audio_data)

        
        if st.button("Test Speaker Recognition"):
            process_audio(temp_audio_path)
        
        # Clean up the temporary file
        os.remove(temp_audio_path)