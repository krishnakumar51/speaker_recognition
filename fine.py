import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
from io import BytesIO
from app import register_authorized_speaker, is_authorized_speaker

# Constants
AUTHORIZED_USER_FOLDER = "authenticated_user"  # Path to the authorized speaker folder
TEMP_AUDIO_FOLDER = "temp_audio"  # Path to save uploaded and recorded audio files temporarily
SAMPLE_RATE = 22050  # Sample rate for recording
DURATION = 5  # Duration of the recording in seconds

# Create temp_audio directory if it doesn't exist
if not os.path.exists(TEMP_AUDIO_FOLDER):
    os.makedirs(TEMP_AUDIO_FOLDER)

# Initialize session state
if 'authorized_embedding_avg' not in st.session_state:
    st.session_state.authorized_embedding_avg = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

# Register the authorized speaker (this can be done once, or dynamically)
if os.path.exists(AUTHORIZED_USER_FOLDER) and st.session_state.authorized_embedding_avg is None:
    st.session_state.authorized_embedding_avg = register_authorized_speaker(AUTHORIZED_USER_FOLDER)
    st.success("Authorized speaker registered!")

def save_audio_file(audio_data, file_name):
    file_path = os.path.join(TEMP_AUDIO_FOLDER, file_name)
    sf.write(file_path, audio_data, SAMPLE_RATE)
    return file_path

def play_audio(file_path):
    """Play audio file in Streamlit."""
    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

def generate_simulated_speech(duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Generate a simulated speech-like audio signal.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a complex tone (fundamental frequency + harmonics)
    fundamental_freq = 150  # Approximate fundamental frequency of human voice
    audio = np.sin(2 * np.pi * fundamental_freq * t)
    audio += 0.5 * np.sin(2 * np.pi * 2 * fundamental_freq * t)  # First harmonic
    audio += 0.3 * np.sin(2 * np.pi * 3 * fundamental_freq * t)  # Second harmonic
    
    # Add some amplitude variation to simulate speech cadence
    envelope = np.sin(2 * np.pi * 0.5 * t) ** 2 + 0.5
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio

def record_audio():
    """
    Simulate recording audio.
    """
    st.session_state.recording = True
    st.session_state.audio_data = generate_simulated_speech()

def stop_recording():
    """
    Stop the audio recording simulation.
    """
    st.session_state.recording = False

# Handle file uploads or recording
st.header("Test the Speaker Recognition System")
option = st.selectbox("Choose an option", ("Upload an audio file", "Record your voice"))

if option == "Upload an audio file":
    uploaded_file = st.file_uploader("Upload an audio file to test", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Load the audio file using librosa
        audio_data, _ = librosa.load(uploaded_file, sr=SAMPLE_RATE)
        
        # Save the uploaded file to a temporary path
        new_audio_path = save_audio_file(audio_data, uploaded_file.name)
        
        if st.session_state.authorized_embedding_avg is not None:
            is_authorized, similarity = is_authorized_speaker(new_audio_path, st.session_state.authorized_embedding_avg)
            
            if is_authorized:
                st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
            else:
                st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")
        
        # Playback
        play_audio(new_audio_path)
        
        # Clean up the temporary file
        os.remove(new_audio_path)

elif option == "Record your voice":
    st.write(f"Click 'Start Recording' to begin a {DURATION}-second simulated recording.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Recording"):
            record_audio()
    
    with col2:
        if st.button("Stop Recording"):
            stop_recording()
    
    if st.session_state.recording:
        st.write("Simulating recording... (This would be when you speak)")
        # The actual audio data is generated in the record_audio function
    
    if not st.session_state.recording and st.session_state.audio_data is not None:
        st.write("Simulated recording finished. Testing the audio...")
        
        # Save the simulated audio
        temp_audio_path = save_audio_file(st.session_state.audio_data, "simulated_audio.wav")
        
        # Play back the simulated audio
        st.audio(temp_audio_path, format='audio/wav')
        
        # Perform speaker recognition
        if st.session_state.authorized_embedding_avg is not None:
            is_authorized, similarity = is_authorized_speaker(temp_audio_path, st.session_state.authorized_embedding_avg)
            
            if is_authorized:
                st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
            else:
                st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")
        
        # Clean up the temporary file
        os.remove(temp_audio_path)
        
        # Reset the audio data
        st.session_state.audio_data = None