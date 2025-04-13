# 🎵 Hybrid Deep Learning-Based Music Recommendation System
![Powered by Spotify](https://img.shields.io/badge/Dataset-Spotify-green?logo=spotify&style=for-the-badge)


This repository presents the code and methodology for a **Hybrid Music Recommendation System**, developed using **Python** and **multi-layer perceptron (MLP) neural networks** in **Jupyter Notebook**. This work was part of my **Master of Engineering thesis in Computer Science** and was also published in an **international conference**.

---

## 📌 Overview

The goal of this project is to provide personalized music recommendations using a hybrid model that combines both **content-based filtering** and **collaborative filtering** techniques with **deep learning**. Our model is built using multiple MLP layers trained on the **Spotify Million Playlist Dataset**, and achieves significantly higher accuracy and precision compared to traditional models.

---

## 🧠 Key Features

- 📁 Built in Python using TensorFlow and Jupyter Notebooks.
- 🤖 Uses a hybrid recommendation approach combining:
  - Content-Based Filtering (song metadata, genres, audio features)
  - Collaborative Filtering (user-item interaction matrix)
- 🔗 Integrates with Spotify Web API via `Spotipy`.
- 🧪 Evaluated with different activation functions (ReLU, sigmoid, softmax, SeLU).
- 📊 Performance metrics: accuracy of **97.5%** and precision of **0.94** with hybrid model.

---

## 📂 Project Structure

- `notebook/`
  - `Hybrid_Music_Recommendation.ipynb` – Main notebook with code and results
- `data/`
  - `spotify_dataset.csv` – Processed Spotify playlist data
- `models/`
  - `mlp_model.h5` – Trained MLP model (optional)
- `README.md` – This file
- `requirements.txt` – Python dependencies

---

## 🏗️ Methodology

### 🔹 Content-Based Filtering
- Uses metadata like **genre** and **release year**.
- Genres are clustered using **K-Means** based on audio features.
- Recommends songs that are similar in metadata to the user's past preferences.

### 🔹 Collaborative Filtering
- Builds a **sparse user-item matrix** (playlist-track associations).
- Learns **latent embeddings** for users and tracks.
- Applies MLP on concatenated embeddings to model interactions.

### 🔹 Hybrid Model
- Combines both filtering techniques as **input to a deep neural network**.
- Final MLP output gives a **probability score (0 to 1)** for song recommendations.

---

## 🧪 Results

| Model                        | Accuracy | Precision |
|-----------------------------|----------|-----------|
| Content-Based (MLP)         | 63.5%    | 0.67      |
| Collaborative Filtering     | 75%      | 0.81      |
| **Hybrid Model (MLP)**      | **97.5%**| **0.94**   |

### 🔍 Activation Function Comparison

| Hidden Layer + Output Layer | Accuracy | Precision |
|-----------------------------|----------|-----------|
| ReLU + Sigmoid              | 98%      | 0.94      |
| SeLU + Sigmoid              | 85%      | 0.82      |
| ReLU + Softmax              | 20.5%    | 0.2       |
| SeLU + Softmax              | 19%      | 0.19      |

---

## 📚 Publication

This research was published at the **International Conference on Intelligent Computing and Big Data Analytics (ICIB)**.

**Title**: *Hybrid Deep Learning-Based Music Recommendation System*  
**Authors**: M. Sunitha, Dr. T. Adilakshmi, Mehar Unissa  
**Affiliation**: Vasavi College of Engineering, Hyderabad, India

---

## 🚀 Future Work

- 🎭 Add facial emotion detection to recommend music based on mood.
- 📱 Build a mobile UI for real-time personalized playlists.
- 🧩 Integrate user behavior signals like skips and replays for improved recommendations.

---

## ⚙️ Requirements

```bash
Python 3.8+
TensorFlow
Pandas
NumPy
Scikit-learn
Spotipy
Matplotlib
```
## 🙌 Acknowledgements
This work was supervised by Dr. T. Adilakshmi and supported by the faculty at Vasavi College of Engineering.</br>
Special thanks to the amazing team at [Spotify](https://github.com/spotify) for providing the [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge), which was essential for building and evaluating this music recommendation system.


