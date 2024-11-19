import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, SpatialDropout1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)

# Load and filter dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['emotion'].isin(['happiness', 'sadness'])]  # Filter for happiness and sadness
    return df

# Clean text function
def clean_text(text):
    text = tf.strings.regex_replace(text, r"http\S+", "")  # Remove URLs
    text = tf.strings.regex_replace(text, r"[^a-zA-Z\s]", "")  # Remove symbols and punctuation
    text = tf.strings.lower(text)  # Convert to lowercase
    return text

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

# Load dataset
file_path = "tweet_emotions.csv"  # Replace with your dataset path
df = load_data(file_path)

# Prepare data
X = df['content']  # Tweet text
y = df['emotion'].astype('category').cat.codes  # Convert emotions to numeric codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Define TextVectorization layer
vectorizer = TextVectorization(max_tokens=5000, output_sequence_length=256, standardize=clean_text)
vectorizer.adapt(X_train)

# Define the model
model = Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    Embedding(input_dim=5000, output_dim=128),
    SpatialDropout1D(0.2),
    GlobalMaxPooling1D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=256,
    epochs=20,
    class_weight=class_weights,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot training history
plot_history(history)

# Make predictions
y_pred = model.predict(X_test).argmax(axis=1)

# Classification report
print(classification_report(y_test, y_pred, target_names=['happiness', 'sadness']))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Test on new sentences
test_sentences = [
    "I wish you didn't have to go... everything is so much brighter when you are around.",
    "This is amazing, I love it.",
    "I am done with this.",
    "Meet me when you get back; I love being around you.",
    "This is the worst trash I have tasted."
]
preds = model.predict(test_sentences).argmax(axis=1)
print("Predicted Emotions:", [df['emotion'].cat.categories[pred] for pred in preds])
