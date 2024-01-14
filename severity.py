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
