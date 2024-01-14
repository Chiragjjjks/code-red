import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import glob
import parselmouth
from parselmouth.praat import call

# Define the voice function here
def voice(wave_file):
  def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID)
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0, 0, unit)
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

  file_list = []
  mean_F0_list = []
  sd_F0_list = []
  hnr_list = []
  localJitter_list = []
  localabsoluteJitter_list = []
  rapJitter_list = []
  ppq5Jitter_list = []
  ddpJitter_list = []
  localShimmer_list = []
  localdbShimmer_list = []
  apq3Shimmer_list = []
  aqpq5Shimmer_list = []
  apq11Shimmer_list = []
  ddaShimmer_list = []

    # Go through all the wave files in the folder and measure pitch
  sound = parselmouth.Sound(wave_file)
  (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
  file_list.append(wave_file)
  mean_F0_list.append(meanF0)
  sd_F0_list.append(stdevF0)
  hnr_list.append(hnr)
  localJitter_list.append(localJitter)
  localabsoluteJitter_list.append(localabsoluteJitter)
  rapJitter_list.append(rapJitter)
  ppq5Jitter_list.append(ppq5Jitter)
  ddpJitter_list.append(ddpJitter)
  localShimmer_list.append(localShimmer)
  localdbShimmer_list.append(localdbShimmer)
  apq3Shimmer_list.append(apq3Shimmer)
  aqpq5Shimmer_list.append(aqpq5Shimmer)
  apq11Shimmer_list.append(apq11Shimmer)
  ddaShimmer_list.append(ddaShimmer)
  df = pd.DataFrame(np.column_stack([localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, ddaShimmer_list, hnr_list]),
                  columns=['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA', 'HNR'])
  df.to_csv("shaa.csv", index=False)
  array = np.array([df['Jitter(%)'], df['Jitter(Abs)'], df['Jitter:RAP'], df['Jitter:PPQ'],
       df['Jitter:DDP'], df['Shimmer'], df['Shimmer(dB)'], df['Shimmer:APQ3'], df['Shimmer:APQ5'],
       df['Shimmer:DDA'],  df['HNR']])
  return array

df = pd.read_excel('jeevannew.xlsx')

# Select features and target variables
X = df[['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ',
       'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:DDA', 'HNR']]
y_classification = df['status']
y_severity = df['severity']

# Normalize the input features using min-max scaling
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Split the data into training and testing sets
X_train, X_test, y_classification_train, y_classification_test, y_severity_train, y_severity_test = train_test_split(
    X, y_classification, y_severity, test_size=0.2, random_state=42
)

# Model architecture
input_classification = Input(shape=(X_train.shape[1],))
input_severity = Input(shape=(X_train.shape[1],))

shared_layer = Dense(128, activation='relu')(input_classification)
shared_layer = Dropout(0.5)(shared_layer)

shared_layer = Dense(64, activation='relu')(shared_layer)
shared_layer = Dropout(0.5)(shared_layer)

# Classification branch with sigmoid activation
classification_branch = Dense(1, activation='sigmoid', name='classification_output')(shared_layer)

# Severity branch with linear activation
severity_branch = Dense(1, activation='linear', name='severity_output')(shared_layer)

model = Model(inputs=[input_classification, input_severity], outputs=[classification_branch, severity_branch])

# Compile the model
model.compile(
    optimizer=Adam(lr=0.0001),
    loss={'classification_output': 'binary_crossentropy', 'severity_output': 'mean_squared_error'},
    loss_weights={'classification_output': 0.5, 'severity_output': 0.5}
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training the model
history = model.fit(
    x=[X_train, X_train],
    y={'classification_output': y_classification_train, 'severity_output': y_severity_train},
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Print training history
print("Training History:")
print(history.history)

# Evaluate the model on the test set
loss, classification_loss, severity_loss = model.evaluate(
    x=[X_test, X_test],
    y={'classification_output': y_classification_test, 'severity_output': y_severity_test}
)
print(f"Total Loss: {loss}, Classification Loss: {classification_loss}, Severity Loss: {severity_loss}")

# Example audio features
audio_features = voice('C:/Users/chira/Desktop/bmsit/voice/BL01 ENSS.wav')

# Prediction function
def predict_classification_and_severity(model, audio_features):
    audio_features = np.expand_dims(audio_features, axis=0)
    predictions = model.predict(x=[audio_features, audio_features])
    classification = np.round(predictions[0]).astype(int)
    severity = predictions[1][0]
    return classification, severity

# Make a prediction
classification, severity = predict_classification_and_severity(model, audio_features)
if severity > 1:
    classification = 1

print(f"Classification: {classification}, Severity: {severity}")

##output=Classification: 1, Severity: [2.755847]


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import glob
import parselmouth
from parselmouth.praat import call

# ... (Previous code remains unchanged)

# Make a prediction
classification, severity = predict_classification_and_severity(model, audio_features)

if severity > 3.00:
    print("In advanced stages of dysphagia, patients should be closely monitored for aspiration risk and may require thickened liquids or pureed foods. Consultation with a speech therapist and healthcare provider is crucial for tailored management and support.")
elif severity > 2.00:
    print("Stick to a soft or pureed diet to ease swallowing for mild dysphagia. Sip liquids with meals and maintain an upright posture to prevent choking and aspiration.")
elif severity > 1.00:
    print("Consume smaller, more frequent meals to reduce the risk of choking. Stay hydrated by sipping liquids slowly and using thickened liquids if recommended by a healthcare professional.")
else:
    print("Severity is within normal range.")

print(f"Classification: {classification}, Severity: {severity}")
