import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import soundfile as sf
import tempfile

# Load the trained model and scaler
loaded_svm_model = joblib.load("svm_model_3sec_allF.pkl")
loaded_scaler = joblib.load("scaler_3sec_allF.pkl")

def preprocess_audio(audio_path, target_sr=16000):
    data, original_sr = librosa.load(audio_path, sr=target_sr)
    trimmed_data, _ = librosa.effects.trim(data)
    return trimmed_data

def segment_audio(audio, segment_size=3, overlap=0.5, sr=16000):
    segment_size_samples = int(segment_size * sr)
    hop_length = int(segment_size_samples * (1 - overlap))
    segments = [audio[i:i + segment_size_samples] for i in range(0, len(audio) - segment_size_samples + 1, hop_length)]
    return segments if segments else [audio]

def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    energy = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    return np.concatenate([mfcc_mean, [energy, zcr]])

def predict_emotion_segments(segments, model, scaler):
    predictions = []
    for idx, segment in enumerate(segments):
        features = extract_features(segment)
        standardized_features = scaler.transform(features.reshape(1, -1))
        predicted_label = model.predict(standardized_features)[0]
        predictions.append((f"Segment {idx+1}", predicted_label))
    return predictions

st.title("🎵 Emotion Recognition from Audio")
st.write("Upload an audio file, and the model will predict emotions for each segment.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name
    
    preprocessed_audio = preprocess_audio(temp_filename)
    segments = segment_audio(preprocessed_audio)
    predicted_labels = predict_emotion_segments(segments, loaded_svm_model, loaded_scaler)
    
    # Convert predictions to DataFrame and display as a table
    df = pd.DataFrame(predicted_labels, columns=["Segment", "Predicted Emotion"])
    st.write("### Predicted Emotions per Segment")
    st.dataframe(df)
