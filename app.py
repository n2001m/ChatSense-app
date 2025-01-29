import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load models
loaded_svm_model = joblib.load("svm_model_3sec_allF.pkl")
loaded_scaler = joblib.load("scaler_3sec_allF.pkl")

# Function to preprocess audio
def preprocess_audio(audio_path, target_sr=16000):
    data, original_sr = librosa.load(audio_path, sr=target_sr)
    trimmed_data, _ = librosa.effects.trim(data)
    stft = librosa.stft(trimmed_data)
    noise_estimation = np.mean(np.abs(stft), axis=1)
    clean_stft = np.maximum(np.abs(stft) - 2 * noise_estimation[:, np.newaxis], 0.0)
    clean_data = librosa.istft(clean_stft)
    return clean_data

# Function to segment audio with overlap
def segment_audio(audio, segment_size=3, overlap=0.5):
    segment_size_samples = int(segment_size * 16000)  # Convert segment size to samples
    hop_length = int(segment_size_samples * (1 - overlap))  # Calculate hop length
    if len(audio) <= segment_size_samples:
        segments = [audio]
    else:
        segments = []
        for i in range(0, len(audio) - segment_size_samples + 1, hop_length):
            segment = audio[i:i + segment_size_samples]
            segments.append(segment)
    return segments

# Function to extract features
def extract_features(audio):
    sampling_rate = 16000
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate)
    mfcc_mean = np.mean(mfcc, axis=1)
    speech_rate = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)
    speech_rate_mean = np.mean(speech_rate)
    energy = librosa.feature.rms(y=audio)
    energy_mean = np.mean(energy)
    pitch = librosa.yin(y=audio, fmin=8, fmax=600)
    pitch_mean = np.mean(pitch)
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate)
    kurtosis = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))
    kurtosis_mean = np.mean(kurtosis)
    return np.concatenate([mfcc_mean, [speech_rate_mean, energy_mean, pitch_mean, zcr_mean, kurtosis_mean]])

# Function to predict emotion label for each segment
def predict_emotion_segments(segments, loaded_svm_model, loaded_scaler):
    predicted_labels = []
    for segment in segments:
        segment_features = extract_features(segment)
        standardized_features = loaded_scaler.transform(segment_features.reshape(1, -1))
        predicted_label = loaded_svm_model.predict(standardized_features)
        predicted_labels.append(predicted_label)
    return predicted_labels

# Streamlit App
st.title('Emotion Recognition from Audio')
st.write("Record an audio clip and we'll predict the emotions for each segment.")

# Class to handle the audio transformer
class AudioTransformer(VideoTransformerBase):
    def __init__(self):
        self.audio_data = None

    def transform(self, frame):
        # Capture audio frame when the recording starts
        if frame is not None:
            self.audio_data = frame
        return frame

# Initialize the webrtc_streamer with AudioTransformer
audio_transformer = AudioTransformer()
webrtc_streamer(key="audio-recorder", video_processor_factory=lambda: audio_transformer)

# Record audio functionality
if st.button('Start Recording'):
    if audio_transformer.audio_data is not None:
        # Save audio data to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        with open(temp_file.name, "wb") as f:
            f.write(audio_transformer.audio_data)

        # Preprocess audio and make predictions
        preprocessed_audio = preprocess_audio(temp_file.name)
        segments = segment_audio(preprocessed_audio)
        predicted_labels = predict_emotion_segments(segments, loaded_svm_model, loaded_scaler)
        predicted_labels = [label[0] for label in predicted_labels]

        # Display results
        st.write("Predicted emotions for each segment:")
        for i, label in enumerate(predicted_labels):
            st.write(f"Segment {i+1}: {label}")

        # Option to play back the recorded audio
        st.audio(temp_file.name)
