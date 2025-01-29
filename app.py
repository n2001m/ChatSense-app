import streamlit as st
import numpy as np
import joblib
import librosa
import soundfile as sf
import av
import warnings
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Load the SVM model and scaler
loaded_svm_model = joblib.load("svm_model_3sec_allF.pkl")
loaded_scaler = joblib.load("scaler_3sec_allF.pkl")

st.title("🎤 Audio Emotion Detection App")

# Allow user to choose between uploading or recording audio
option = st.radio("Choose an option:", ["Upload Audio", "Record Audio"])

# Function to preprocess audio
def preprocess_audio(audio_data, sample_rate=16000):
    trimmed_data, _ = librosa.effects.trim(audio_data)
    stft = librosa.stft(trimmed_data)
    noise_estimation = np.mean(np.abs(stft), axis=1)
    clean_stft = np.maximum(np.abs(stft) - 2 * noise_estimation[:, np.newaxis], 0.0)
    clean_data = librosa.istft(clean_stft)
    return clean_data

# Function to segment audio
def segment_audio(audio, segment_size=3, overlap=0.5, sr=16000):
    segment_size_samples = int(segment_size * sr)
    hop_length = int(segment_size_samples * (1 - overlap))
    if len(audio) <= segment_size_samples:
        return [audio]
    return [audio[i:i + segment_size_samples] for i in range(0, len(audio) - segment_size_samples + 1, hop_length)]

# Function to extract features
def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    speech_rate = librosa.feature.spectral_centroid(y=audio, sr=sr)
    speech_rate_mean = np.mean(speech_rate)
    energy = librosa.feature.rms(y=audio)
    energy_mean = np.mean(energy)
    pitch = librosa.yin(y=audio, fmin=8, fmax=600)
    pitch_mean = np.mean(pitch)
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    kurtosis = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))
    kurtosis_mean = np.mean(kurtosis)
    return np.concatenate([mfcc_mean, [speech_rate_mean, energy_mean, pitch_mean, zcr_mean, kurtosis_mean]])

# Function to predict emotion for each segment
def predict_emotion_segments(segments):
    predicted_labels = []
    for segment in segments:
        features = extract_features(segment)
        standardized_features = loaded_scaler.transform(features.reshape(1, -1))
        predicted_label = loaded_svm_model.predict(standardized_features)
        predicted_labels.append(predicted_label[0])
    return predicted_labels

# Function to display waveform
def display_waveform(audio_data, sample_rate):
    fig, ax = plt.subplots()
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

# --- OPTION 1: UPLOAD AUDIO ---
if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")

        # Load and process audio
        with sf.SoundFile(uploaded_file) as f:
            audio_data = f.read(dtype="float32")
            sample_rate = f.samplerate

        # Preprocess and segment audio
        preprocessed_audio = preprocess_audio(audio_data, sample_rate)
        segments = segment_audio(preprocessed_audio)

        # Predict emotions
        predicted_labels = predict_emotion_segments(segments)

        # Display results
        display_waveform(preprocessed_audio, sample_rate)
        st.write("Predicted Emotions:", predicted_labels)

# --- OPTION 2: RECORD AUDIO ---
elif option == "Record Audio":
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame):
            self.frames.append(frame.to_ndarray())
            return frame

    ctx = webrtc_streamer(key="audio-recorder", audio_processor_factory=AudioProcessor)

    if ctx.audio_processor:
        if st.button("Stop Recording & Process"):
            audio_frames = ctx.audio_processor.frames
            if len(audio_frames) > 0:
                audio_data = np.concatenate(audio_frames, axis=0).flatten()
                sample_rate = 44100

                # Preprocess and segment audio
                preprocessed_audio = preprocess_audio(audio_data, sample_rate)
                segments = segment_audio(preprocessed_audio)

                # Predict emotions
                predicted_labels = predict_emotion_segments(segments)

                # Display results
                display_waveform(preprocessed_audio, sample_rate)
                buffer = BytesIO()
                sf.write(buffer, preprocessed_audio, sample_rate, format="wav")
                st.audio(buffer, format="audio/wav")
                st.write("Predicted Emotions:", predicted_labels)
