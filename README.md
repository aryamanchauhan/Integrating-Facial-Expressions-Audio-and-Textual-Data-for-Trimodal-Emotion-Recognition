## **Usage**

1.	Feature Extraction and Data Loading : 
extract_feature(file_name, mfcc, chroma, mel): Extracts features from an audio file based on the parameters provided.
load_data(test_size): Loads audio files, extracts features, and splits data into training and testing sets.

2.	Training the MLP Classifier : 
The MLPClassifier is defined and trained on the extracted features.

3.	Predicting Emotions : 
	The trained model predicts emotions from test data (*test.wav*).
	The results are saved in *predictionfinal.csv*.



## **Text Emotion Recognition**
1.	Data Preparation : 
	Load and filter the dataset for specific emotions.
	Split the data into training and testing sets.

2.	Text Vectorization and Model Definition : 
	Define a text cleaning function.
	Use TextVectorization for text data preprocessing.
	Define the neural network model architecture.

3.	Model Training and Evaluation : 
	Train the model with early stopping.
	Evaluate the model on test data.
	Plot training and validation accuracy and loss.

4.	Making Predictions : 
	Make predictions on new test sentences.
	Print the predicted emotions.




## **Emotion Detection from Video**
1.	Setup : 
	Load a pre-trained face detection classifier and an emotion classification model.
	Define emotion labels.

2.	Video Capture and Processing : 
Open a video capture device (e.g., webcam).
	Continuously process frames to detect faces and classify emotions.
	Update and display emotion predictions in real-time.



## **Results** 

The output of the audio emotion recognition is saved in a CSV file named *predictionfinal.csv*.



## **License**

This project was developed under the guidance of VIT Chennai faculties.



## **Acknowledgments**

This project utilizes the RAVDESS dataset for audio emotion recognition and various libraries for data manipulation, visualization, and neural networks.

