import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# Step 1: Load the Data
def load_dysphonia_data(file_path='final.xlsx'):
    df = pd.read_excel(file_path)
    df = df.drop(df.columns[0],axis=1)
    return df

# Step 2: Preprocess the Data
def preprocess_data(df):
    # Handle missing values or outliers if needed
    # Encode categorical labels
    df['status'] = df['status'].map({'normal': 0, 'affected': 1})
    return df

# Step 3: Split the Data
def split_data(df):
    X = df.drop(columns=['status', 'severity'])
    y_classification = df['status']
    y_severity = df['severity']
    
    X_train, X_temp, y_class_train, y_class_temp, y_severity_train, y_severity_temp = train_test_split(
        X, y_classification, y_severity, test_size=0.3, random_state=42
    )
    X_val, X_test, y_class_val, y_class_test, y_severity_val, y_severity_test = train_test_split(
        X_temp, y_class_temp, y_severity_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, y_severity_train, y_severity_val, y_severity_test

# Step 4: Define the Base Model (VGGish through TensorFlow Hub)
def define_base_model(input_shape):
    vggish_model_url = "https://tfhub.dev/google/vggish/1"
    vggish_model = hub.load(vggish_model_url)
    vggish_model.trainable = False  # Freeze the pretrained layers
    return tf.keras.Sequential([vggish_model])

# Step 5: Define the Multitask Model
def define_multitask_model(base_model, num_classes):
    base_model_output_shape = base_model.output_shape[1:]
    
    classification_branch = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    severity_branch = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    combined_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[classification_branch(base_model.output), severity_branch(base_model.output)]
    )
    
    return combined_model

# Step 6: Compile the Model
def compile_model(model):
    model.compile(
        optimizer='adam',
        loss={'sequential': 'sparse_categorical_crossentropy', 'sequential_1': 'mean_squared_error'},
        metrics={'sequential': 'accuracy', 'sequential_1': 'mae'}
    )

# Step 7: Train the Model
def train_model(model, X_train, y_class_train, y_severity_train, X_val, y_class_val, y_severity_val, epochs=10, batch_size=32):
    model.fit(
        X_train, {'sequential': y_class_train, 'sequential_1': y_severity_train},
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, {'sequential': y_class_val, 'sequential_1': y_severity_val})
    )


    # Step 8: Prediction function
def predict_sample(model, sample):
    # Assuming 'sample' is a single input sample of the same format as the training data
    sample = np.expand_dims(sample, axis=0)
    predictions = model.predict(sample)
    
    # Extracting classification and severity predictions
    classification_prediction = np.argmax(predictions[0])
    severity_prediction = predictions[1][0][0]
    
    #return classification_prediction, severity_prediction
    print(classification_prediction,severity_prediction)
