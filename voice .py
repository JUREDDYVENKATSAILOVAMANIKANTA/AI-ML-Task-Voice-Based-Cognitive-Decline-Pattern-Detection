#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install whisper')


# In[1]:


import whisper
model = whisper.load_model("base")
result = model.transcribe("audio_clip.wav")
transcript = result['text']


# In[3]:


def assess_cognitive_risk(audio_path: str) -> dict:
    # Transcribe
    text = whisper_transcribe(audio_path)
    # Extract features
    features = extract_audio_text_features(audio_path, text)
    # Get anomaly score
    risk_score = model.predict([features])[0]
    return {"risk_score": risk_score, "features": features}


# In[5]:


get_ipython().system('pip install librosa')


# In[ ]:


# Cognitive Impairment Detection via Speech Analysis

import os
import librosa
import numpy as np
import pandas as pd
import whisper
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Load Whisper Model
model = whisper.load_model("base")

# Utility: Transcribe audio to text
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text'], result['segments']

# Utility: Extract features from audio and transcript
def extract_features(audio_path, transcript):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    # Basic audio features
    pitch = librosa.yin(y, fmin=75, fmax=300)
    pitch_std = np.std(pitch)

    # Pause detection (non-silent intervals)
    intervals = librosa.effects.split(y, top_db=20)
    pauses = len(intervals) - 1

    # Text features
    words = transcript.split()
    hesitations = len(re.findall(r'\b(um+|uh+)\b', transcript.lower()))
    speech_rate = len(words) / duration
    pauses_per_sentence = pauses / max(1, transcript.count('.'))

    # Word recall estimate (word repetition/lost)
    unique_words = len(set(words))
    lost_words = len(words) - unique_words

    return {
        "hesitations": hesitations,
        "pauses_per_sentence": pauses_per_sentence,
        "speech_rate": speech_rate,
        "pitch_std": pitch_std,
        "lost_words": lost_words
    }

# Analyze multiple audio files
def analyze_directory(audio_dir):
    records = []
    for fname in os.listdir(audio_dir):
        if fname.endswith(".wav"):
            path = os.path.join(audio_dir, fname)
            transcript, _ = transcribe_audio(path)
            features = extract_features(path, transcript)
            features["file"] = fname
            records.append(features)
    return pd.DataFrame(records)

# Unsupervised anomaly detection
def detect_anomalies(df):
    feature_cols = [col for col in df.columns if col != 'file']
    X = StandardScaler().fit_transform(df[feature_cols])
    clf = IsolationForest(contamination=0.2, random_state=42)
    df["risk_score"] = clf.fit_predict(X)  # -1 = anomaly

    # Visualize with PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], c=df['risk_score'], cmap='coolwarm')
    plt.title("Anomaly Detection (Cognitive Risk)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='Risk Score')
    plt.show()
    return df

# API-Ready Function
def assess_cognitive_risk(audio_path):
    transcript, _ = transcribe_audio(audio_path)
    features = extract_features(audio_path, transcript)
    return features

# Example usage
if __name__ == "__main__":
    audio_dir = "./audio_samples"  # Folder with .wav files
    df = analyze_directory(audio_dir)
    result_df = detect_anomalies(df)
    print(result_df.sort_values("risk_score"))

