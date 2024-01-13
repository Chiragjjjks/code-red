import librosa
import numpy as np
import matplotlib.pyplot as plt

def audio_to_mel_spectrogram(audio_file, n_mels=128, fmax=8000):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmax=fmax)
    return mel_spectrogram

def preprocess_mel_spectrogram(mel_spectrogram):
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    reshaped_log_mel_spectrogram = log_mel_spectrogram.reshape((1,) + log_mel_spectrogram.shape)
    normalized_log_mel_spectrogram = reshaped_log_mel_spectrogram / 255.0
    return normalized_log_mel_spectrogram

# Example usage
audio_file_path = 'voice/Sj30002_E_NSS.wav'  # Replace with the actual path to your audio file

# Call the audio_to_mel_spectrogram function
mel_spectrogram = audio_to_mel_spectrogram(audio_file_path)

# Call the preprocess_mel_spectrogram function
normalized_log_mel_spectrogram = preprocess_mel_spectrogram(mel_spectrogram)

# Plot the mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()

# Plot the normalized log mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(normalized_log_mel_spectrogram[0], y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Normalized Log Mel Spectrogram')
plt.show()
