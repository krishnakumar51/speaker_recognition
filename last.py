import streamlit as st
import numpy as np
import torchaudio
import os
import io
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

# Record audio using JavaScript
st.markdown(
    """
    <script>
    let mediaRecorder;
    let audioChunks = [];

    const startRecording = () => {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                document.getElementById("recorded_audio").src = audioUrl;
                document.getElementById("audio_data").value = audioBlob;
            };
            mediaRecorder.start();
            document.getElementById("status").innerText = "Recording...";
        });
    };

    const stopRecording = () => {
        if (mediaRecorder) {
            mediaRecorder.stop();
            document.getElementById("status").innerText = "Recording stopped.";
        }
    };
    </script>
    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop Recording</button>
    <div id="status"></div>
    <audio id="recorded_audio" controls></audio>
    <input type="hidden" id="audio_data" />
    """,
    unsafe_allow_html=True
)

# Handle the recorded audio
audio_data = st.file_uploader("Upload a recording or use the recorded audio", type=["wav", "mp3"], key="uploader")

if audio_data is not None:
    # Save the uploaded file or recorded audio
    new_audio_path = f"temp_audio/{audio_data.name}"
    with open(new_audio_path, "wb") as f:
        f.write(audio_data.read())

    # Check if the uploaded or recorded file matches the authorized speaker
    if authorized_embedding_avg is not None:
        is_authorized, similarity = is_authorized_speaker(new_audio_path, authorized_embedding_avg)
        
        # Display the results
        if is_authorized:
            st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
        else:
            st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")

    # Clean up the temporary file
    os.remove(new_audio_path)
