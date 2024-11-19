# **Emotion Recognition Suite**

This repository implements multi-modal emotion recognition from **speech**, **text**, and **video** data. Each modality uses tailored preprocessing techniques, neural network architectures, and evaluation methods to achieve robust performance.

---

## **Speech Emotion Recognition**

#### 1. **Feature Extraction and Data Loading**  
- **Functions**:  
  - `extract_feature(file_name, mfcc, chroma, mel)`: Extracts features like MFCC, Chroma, and Mel Spectrogram from audio files.  
  - `load_data(test_size)`: Loads audio files, extracts features, and splits data into training and testing sets.  

#### 2. **Training the MLP Classifier**  
- Define the `MLPClassifier` with the following parameters:  
  - Hidden layer sizes: `(300,)`  
  - Batch size: `256`  
  - Learning rate: `adaptive`  
  - Maximum iterations: `500`  
- Train the model using the extracted features.

#### 3. **Predicting Emotions**  
- Predict emotions for new test audio files (e.g., `test.wav`) using the trained model.  
- Save prediction results, including probabilities, in `predictionfinal.csv`.

---

## **Text Emotion Recognition**

#### 1. **Data Preparation**  
- **Dataset Loading**: Load and filter data to include only the emotions `happiness` and `sadness`.  
- **Data Splitting**: Split data into training and testing sets using an 80-20 split (`random_state=42`).  
- **Class Balancing**: Compute class weights to handle imbalanced classes.  

#### 2. **Text Preprocessing and Vectorization**  
- Define a text cleaning function to:  
  - Convert text to lowercase.  
  - Remove URLs, symbols, and punctuation.  
- Use a `TextVectorization` layer with:  
  - Maximum tokens: `5000`  
  - Output sequence length: `256`  
  - Custom standardization using the cleaning function.  

#### 3. **Model Definition and Training**  
- Define a neural network model architecture with:  
  - **Embedding Layer**: Converts text tokens to dense embeddings.  
  - **Spatial Dropout1D**: For regularization.  
  - **GlobalMaxPooling1D**: Extracts features from the embedding.  
  - **Dense Layers**: Hidden layer with 256 units and output layer with softmax activation for classification.  
- Compile the model with a suitable optimizer (e.g., Adam) and loss function (e.g., categorical cross-entropy).  
- Train the model using early stopping to prevent overfitting.  

#### 4. **Evaluation and Predictions**  
- Evaluate the model on the test set and calculate metrics such as accuracy, precision, recall, and F1-score.  
- Plot training and validation accuracy/loss curves.  
- Make predictions on test sentences and print predicted emotions.  

---

## **Video Emotion Recognition**

#### 1. **Setup and Initialization**  
- Load a pre-trained face detection classifier (e.g., `haarcascade_frontalface_default.xml`).  
- Load an emotion classification model (e.g., `model.h5`).  
- Define emotion labels: `['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']`.  

#### 2. **Real-Time Video Processing**  
- Open a video capture device (e.g., webcam).  
- For each frame:  
  - Convert the frame to grayscale.  
  - Detect faces using the Haar Cascade classifier.  
  - Preprocess each detected face (e.g., resize to match model input size).  

#### 3. **Emotion Classification**  
- Predict emotions for detected faces using the classification model.  
- Update a live plot showing probabilities for each emotion label.  
- Display the emotion label on the video feed.  

#### 4. **Visualization and Termination**  
- Create subplots for real-time emotion probabilities using Matplotlib.  
- Continuously update the plot and video feed until the user exits.  
- Close the video capture device and destroy all windows.


## **Results**
Each modality demonstrates reliable performance, showcasing its potential for integration into emotion-aware applications. This repository provides a comprehensive foundation for multi-modal emotion recognition.  
