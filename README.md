# ECG Heartbeat Classification using Hybrid CNN-LSTM Model

### Project Overview

This project implements a deep learning approach for classifying ECG heartbeats using a hybrid CNN-LSTM architecture. The model is trained on the MIT-BIH Arrhythmia Database to distinguish between five types of heartbeats:

Non-Ectopic (Normal) beats
Unknown beats
Ventricular ectopic beats
Supraventricular ectopic beats
Fusion beats
Key Features

### Data Handling:
Loads and preprocesses ECG data from MIT-BIH database
Handles class imbalance through upsampling of minority classes
Provides visualization tools for ECG signals and class distribution

### Model Architecture:
Hybrid CNN-LSTM network combining convolutional and recurrent layers
Multiple Conv1D layers with LeakyReLU activation
LSTM layers for temporal pattern recognition
Dropout layers for regularization
Training & Evaluation:
Adam optimizer with learning rate 0.001
Early stopping to prevent overfitting
Comprehensive evaluation metrics including:
Accuracy plots
Classification reports
Confusion matrix visualization
File Structure

ECG_hybrid_model.py: Main implementation file containing the ECGhybrid class
MIT-BIH dataset files (CSV format):
mitbih_train.csv
mitbih_test.csv
