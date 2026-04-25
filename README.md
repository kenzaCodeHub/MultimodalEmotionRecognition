# Multimodal Emotion Recognition — Big Data Project

A project combining **text** and **image** modalities for emotion classification, then analyzing the alignment between both predictions using statistical correlation.

## Objective

Explore how emotions are conveyed through two different channels — written text and facial expressions — and measure the degree of alignment between them using Pearson and Spearman correlations.

## Approach

### Text Pipeline
- Dataset: ~416K labeled text samples (CSV)
- Preprocessing: URL removal, lowercasing, tokenization, stopword removal, lemmatization (NLTK)
- Vectorization: TF-IDF
- Model: Logistic Regression (scikit-learn)

### Image Pipeline
- Dataset: FER2013 — 28,709 grayscale facial expression images (48×48), loaded via DeepLake
- Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Preprocessing: resizing, normalization, channel expansion
- Model: CNN with 3 convolutional blocks + dropout (TensorFlow/Keras)

### Cross-Modal Analysis
- Predictions from both models are aligned and compared
- Pearson and Spearman correlation coefficients
- Visualizations: heatmap, grouped countplots, distribution charts

## Results

| Model | Task | Accuracy |
|---|---|---|
| Logistic Regression | Text emotion classification | **89.4%** |
| CNN (3 conv layers) | Image emotion classification | ~58% |

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, NLTK, DeepLake, Seaborn, Matplotlib, SciPy

## Dataset Sources

- **Text**: [Emotion Dataset](https://www.kaggle.com/)
- **Images**: [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/)

