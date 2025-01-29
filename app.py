import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import joblib
import tempfile
import wave

# Load SVM model and scaler
loaded_svm_model = joblib.load("svm_model_3sec_allF.pkl")
loaded_scaler = joblib.load("scaler_3sec_allF.pkl")


# Function to preprocess audio
def preprocess_audio(data, sr=16000):
    trimmed_data, _ = librosa.effects.trim(data)
    return trimmed_data


# Function to segment audio
def segment_audio(audio, sr=16000, segment_size=3, overlap=0.5):
    segment_size_samples = int(segment_size * sr)
    hop_length = int(segment_size_samples * (1 - overlap))

    if len(audio) <= segment_size_samples:
        return [audio]

    segments = [audio[i:i + segment_size_samples]
                for i in range(0, len(audio) - segment_size_samples + 1, hop_length)]
    return segments


# Function to extract features
def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    energy = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    return np.concatenate([mfcc_mean, [energy, zcr]])


# Function to predict emotion
def predict_emotion(segments):
    predicted_labels = []
    for segment in segments:
        features = extract_features(segment)
        standardized_features = loaded_scaler.transform(features.reshape(1, -1))
        predicted_label = loaded_svm_model.predict(standardized_features)
        predicted_labels.append(predicted_label[0])
    return predicted_labels


# Streamlit UI
st.title("Real-Time Audio Emotion Recognition")
st.write("Record an audio clip and get emotion predictions for each segment.")

# Record Audio
duration = st.slider("Select Recording Duration (seconds)", 1, 10, 3)
if st.button("Record Audio"):
    st.write("Recording...")
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording complete!")

    # Process and Predict
    audio_data = recording.flatten()
    preprocessed_audio = preprocess_audio(audio_data, sr=fs)
    segments = segment_audio(preprocessed_audio, sr=fs)
    predicted_labels = predict_emotion(segments)

    # Display results
    st.write("Predicted Emotions for each segment:")
    st.table({"Segment": list(range(1, len(predicted_labels) + 1)), "Emotion": predicted_labels})
