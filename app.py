import os
import librosa
import torch
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained ECAPA-TDNN model
spk_rec_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                                savedir="pretrained_models/spkrec-ecapa-voxceleb")

# Function to extract embeddings from audio using librosa
def get_embedding(audio_path):
    # Load audio using librosa
    signal, fs = librosa.load(audio_path, sr=None)
    
    # Convert the numpy array to a torch tensor and add a batch dimension
    signal = torch.tensor(np.expand_dims(signal, axis=0))  # Shape: (1, waveform_length)
    
    # Get the embeddings using the model
    embeddings = spk_rec_model.encode_batch(signal)
    
    # Return the embeddings as a numpy array
    return embeddings.squeeze().cpu().detach().numpy()

# Function to register an authorized speaker (average of multiple audio samples)
def register_authorized_speaker(authorized_folder):
    authorized_embeddings = []
    
    # Iterate over all audio files in the authorized speaker's folder
    for file_name in os.listdir(authorized_folder):
        file_path = os.path.join(authorized_folder, file_name)
        embedding = get_embedding(file_path)
        authorized_embeddings.append(embedding)
    
    # Average the embeddings to get a single "registered" embedding for the authorized speaker
    authorized_embedding_avg = np.mean(authorized_embeddings, axis=0)
    
    return authorized_embedding_avg

# Function to compare new audio sample against the registered authorized speaker's embedding
def is_authorized_speaker(new_audio_path, authorized_embedding_avg, threshold=0.75):
    new_embedding = get_embedding(new_audio_path)
    
    # Compute cosine similarity between new embedding and the authorized speaker's embedding
    similarity = cosine_similarity([new_embedding], [authorized_embedding_avg])[0][0]
    
    print(f"Cosine Similarity: {similarity}")
    
    # Check if similarity is above the threshold
    if similarity >= threshold:
        return True, similarity
    else:
        return False, similarity
