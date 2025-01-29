import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile

# Load pre-trained models
loaded_svm_model = joblib.load("svm_model_3sec_allF.pkl")
loaded_scaler = joblib.load("scaler_3sec_allF.pkl")

def preprocess_audio(audio_path, target_sr=16000):
    data, _ = librosa.load(audio_path, sr=target_sr)
    trimmed_data, _ = librosa.effects.trim(data)
    return trimmed_data

def segment_audio(audio, segment_size=3, overlap=0.5):
    segment_size_samples = int(segment_size * 16000)
    hop_length = int(segment_size_samples * (1 - overlap))
    if len(audio) <= segment_size_samples:
        return [audio]
    return [audio[i:i + segment_size_samples] for i in range(0, len(audio) - segment_size_samples + 1, hop_length)]

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

def predict_emotion_segments(segments):
    predicted_labels = []
    for segment in segments:
        segment_features = extract_features(segment)
        standardized_features = loaded_scaler.transform(segment_features.reshape(1, -1))
        predicted_label = loaded_svm_model.predict(standardized_features)
        predicted_labels.append(predicted_label[0])
    return predicted_labels

st.title("Audio Emotion Recognition")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.audio(uploaded_file, format="audio/wav")
    preprocessed_audio = preprocess_audio(tmp_path)
    
    if len(preprocessed_audio) / 16000 < 3:
        audio_features = extract_features(preprocessed_audio)
        standardized_features = loaded_scaler.transform(audio_features.reshape(1, -1))
        predicted_label = loaded_svm_model.predict(standardized_features)[0]
        st.write(f"Predicted Emotion: {predicted_label}")
    else:
        segments = segment_audio(preprocessed_audio)
        predicted_labels = predict_emotion_segments(segments)
        st.write("### Predicted Emotions per Segment:")
        for i, label in enumerate(predicted_labels):
            st.write(f"- Segment {i+1}: {label}")
    os.remove(tmp_path)
