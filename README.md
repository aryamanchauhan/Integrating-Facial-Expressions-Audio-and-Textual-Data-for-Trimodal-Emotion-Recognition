# **Trimodal Emotion Recognition (Real Time)**

## **Speech Emotion Recognition**

#### 1. **Feature Extraction and Data Loading**  
- `extract_feature(file_name, mfcc, chroma, mel)`: Extracts audio features like MFCC, Chroma, and Mel Spectrogram from an audio file.  
- `load_data(test_size)`: Loads audio files, extracts features, and splits the dataset into training and testing sets.

#### 2. **Training the MLP Classifier**  
- The `MLPClassifier` is defined with parameters like batch size, hidden layer size, and adaptive learning rate.  
- The model is trained on the extracted features.

#### 3. **Predicting Emotions**  
- The trained model predicts emotions for test audio files (e.g., `test.wav`).  
- Prediction results, including probabilities, are saved in `predictionfinal.csv`.  

---

## **Text Emotion Recognition**

#### 1. **Data Preparation**  
- Load and filter the dataset to focus on specific emotions.  
- Split the data into training and testing sets.

#### 2. **Text Vectorization and Model Definition**  
- Define a text cleaning function to preprocess raw text.  
- Use `TextVectorization` for converting text data into numerical form.  
- Define the neural network model architecture.

#### 3. **Model Training and Evaluation**  
- Train the model with early stopping to prevent overfitting.  
- Evaluate the model's performance on test data.  
- Plot training and validation accuracy and loss curves.

#### 4. **Making Predictions**  
- Predict emotions for new test sentences.  
- Display the predicted emotions.  

---

## **Emotion Detection from Video**

#### 1. **Setup**  
- Load a pre-trained face detection classifier and an emotion classification model.  
- Define emotion labels for classification.  

#### 2. **Video Capture and Processing**  
- Open a video capture device (e.g., webcam).  
- Continuously process video frames to:  
  - Detect faces.  
  - Classify emotions for each detected face.  
- Update and display emotion predictions in real time.  

