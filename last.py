import streamlit as st
import os
from app import register_authorized_speaker, is_authorized_speaker
import io

# Constants
AUTHORIZED_USER_FOLDER = "authenticated_user"  # Path to the authorized speaker folder

# Streamlit app title
st.title("Speaker Recognition App")

# Register the authorized speaker (this can be done once, or dynamically)
authorized_embedding_avg = None
if os.path.exists(AUTHORIZED_USER_FOLDER):
    authorized_embedding_avg = register_authorized_speaker(AUTHORIZED_USER_FOLDER)
    st.success("Authorized speaker registered!")

# HTML and JavaScript for recording audio
st.markdown(
    """
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let audioBlobUrl = "";

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioBlobUrl = URL.createObjectURL(audioBlob);
                document.getElementById("recorded_audio").src = audioBlobUrl;
                document.getElementById("recorded_audio").style.display = "block";
                document.getElementById("audio_data").value = audioBlobUrl;
                document.getElementById("status").innerText = "Recording stopped.";
                audioChunks = [];  // Clear the chunks for the next recording
            };
            mediaRecorder.start();
            document.getElementById("status").innerText = "Recording...";
        }).catch(err => {
            console.error('Error accessing media devices.', err);
            document.getElementById("status").innerText = "Failed to start recording.";
        });
    }

    function stopRecording() {
        if (mediaRecorder) {
            mediaRecorder.stop();
        } else {
            document.getElementById("status").innerText = "No active recording.";
        }
    }

    function uploadAudio() {
        if (audioBlobUrl) {
            const formData = new FormData();
            fetch(audioBlobUrl).then(response => response.blob()).then(blob => {
                formData.append("file", blob, "recording.wav");
                fetch("/upload", {
                    method: "POST",
                    body: formData
                }).then(response => response.json()).then(data => {
                    if (data.success) {
                        document.getElementById("status").innerText = "Audio uploaded successfully.";
                    } else {
                        document.getElementById("status").innerText = "Audio upload failed.";
                    }
                }).catch(err => {
                    console.error('Error uploading audio.', err);
                    document.getElementById("status").innerText = "Error uploading audio.";
                });
            });
        } else {
            document.getElementById("status").innerText = "No audio to upload.";
        }
    }
    </script>
    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop Recording</button>
    <button onclick="uploadAudio()">Upload Recording</button>
    <div id="status"></div>
    <audio id="recorded_audio" controls style="display:none;"></audio>
    <input type="hidden" id="audio_data" />
    """,
    unsafe_allow_html=True
)

# Function to handle file uploads (simulating the upload endpoint)
def handle_upload():
    if st.session_state.get("audio_data"):
        audio_url = st.session_state["audio_data"]
        response = requests.get(audio_url)
        audio_blob = io.BytesIO(response.content)
        new_audio_path = f"temp_audio/recording.wav"
        with open(new_audio_path, "wb") as f:
            f.write(audio_blob.read())

        # Check if the uploaded or recorded file matches the authorized speaker
        if authorized_embedding_avg:
            is_authorized, similarity = is_authorized_speaker(new_audio_path, authorized_embedding_avg)

            # Display the results
            if is_authorized:
                st.success(f"Authorized speaker detected! Similarity score: {similarity:.2f}")
            else:
                st.error(f"Unknown speaker detected. Similarity score: {similarity:.2f}")

        # Clean up the temporary file
        os.remove(new_audio_path)

if st.button("Process Recording"):
    handle_upload()
