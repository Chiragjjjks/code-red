import os
import numpy as np
from scipy.io.wavfile import read
from scipy.stats import entropy
import librosa

def calculate_pitch_period_entropy(audio_path):
    # Load audio file
    audio_signal, _ = librosa.load(audio_path, sr=None)

    # Extract pitch periods using Librosa (replace with your preferred method)
    pitch_periods, _ = librosa.core.piptrack(y=audio_signal)

    # Take the mean along the time axis to get a single pitch period value per frame
    pitch_periods_mean = np.mean(pitch_periods, axis=0)

    # Normalize pitch periods (optional)
    normalized_pitch_periods = (pitch_periods_mean - np.min(pitch_periods_mean)) / (np.max(pitch_periods_mean) - np.min(pitch_periods_mean))

    # Calculate entropy
    pitch_entropy = entropy(normalized_pitch_periods)

    return pitch_entropy

def calculate_jitter(peak):
    differences = np.diff(peak)
    non_zero_diff = differences[differences != 0]
    
    if len(non_zero_diff) > 0:
        sums = np.sum(np.abs(20 * np.log10(np.abs(non_zero_diff))))
        return sums / len(non_zero_diff)
    else:
        return 0.0

def calculate_shimmer(peak):
    peakf = np.abs(np.fft.fft(peak))
    sumps = np.sum((peakf[1:] ** -1) - (peakf[:-1] ** -1))
    return sumps / (len(peakf) - 1)

def calculate_jitterrap(peak):
    sortedp = np.sort(peak)
    sortedf = np.abs(np.fft.fft(sortedp))
    
    dif = np.abs(sortedp[11] - sortedp[15])
    avgabsdiff = dif / 4
    
    avgneigh1 = np.abs(sortedp[6] - sortedp[10])
    avgneigh2 = np.abs(sortedp[17] - sortedp[22])
    avg = (dif + avgneigh1 + avgneigh2) / 3

    suh = np.sum(np.abs(sortedf[11:16] ** -1))
    period = suh / 5

    return (avgabsdiff + avg) / period

def analyze_audio(audio_path):
    sample_rate, audio = read(audio_path)
    peak = audio
    jitter = calculate_jitter(peak)
    shimmer = calculate_shimmer(peak)
    jitterrap = calculate_jitterrap(peak)
    pitch_entropy = calculate_pitch_period_entropy(audio_path)

    return {
        "JITTER": jitter,
        "SHIMMER": shimmer,
        "JITTERRAP": jitterrap,
        "PITCH_ENTROPY": pitch_entropy
    }

def analyze_audio_folder(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    results_dict = {}

    for audio_file in audio_files:
        audio_file_path = os.path.join(folder_path, audio_file)
        analysis_results = analyze_audio(audio_file_path)
        results_dict[audio_file] = analysis_results

    return results_dict

# Example usage with a folder path
audio_folder_path = 'voice'
folder_analysis_results = analyze_audio_folder(audio_folder_path)

# Print or use the calculated features for each file
for file_name, analysis_results in folder_analysis_results.items():
    print(f"\nAnalysis results for {file_name}:")
    for feature, value in analysis_results.items():
        print(f"{feature}: {value}")
