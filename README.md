Multimodal Emotion Recognition — Big Data Project

A project combining text and image modalities for emotion classification. The text pipeline uses TF-IDF vectorization with Logistic Regression on a CSV emotion dataset. The image pipeline trains a CNN on the FER2013 facial expression dataset (loaded via DeepLake). The two models are then compared using Pearson and Spearman correlation to analyze the alignment between text-predicted and image-predicted emotions.

Stack: Python, TensorFlow/Keras, scikit-learn, NLTK, DeepLake, Seaborn.
