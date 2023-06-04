import scipy
import librosa
import matplotlib.pyplot as plt
import numpy as np
from hadamard import add_noise, denoise_signal, hadamard_transform, hadamard_matrix, inverse_hadamard_transform

def read_audio_file(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None)
    return signal, sample_rate

def plot_signal_comparison(signal, denoised_signal, sample_rate):
    time = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.plot(time, denoised_signal)
    plt.title('Denoised Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

audio_data, sample_rate = read_audio_file('audio.wav')

normalized_audio = audio_data / np.max(np.abs(audio_data))

target_sample_rate = 44100
resampled_audio = librosa.resample(normalized_audio, sample_rate,
target_sample_rate)

noise_level = 0.5
noisy_signal = add_noise(resampled_audio, noise_level)

frame_length = 1024
hop_length = 512
frames = librosa.util.frame(resampled_audio, frame_length=
frame_length, hop_length=hop_length)

spectral_frames = np.apply_along_axis(hadamard_transform, axis=1, arr=frames)

transformed_signal = hadamard_transform(noisy_signal)
threshold = 0.2
transformed_signal[np.abs(transformed_signal) < threshold] = 0

noise_frames = spectral_frames[:, :frames]
noise_profile = np.mean(noise_frames, axis=1)

denoised_signal = denoise_signal(noisy_signal, noise_level)

noise_mask = np.where(noise_profile > threshold, 0, 1)

denoised_frames = spectral_frames * noise_mask[:, np.newaxis]

reconstructed_frames = inverse_hadamard_transform(denoised_frames, axis=1)

reconstructed_audio = librosa.util.frame(reconstructed_frames, frame_length=frame_length, hop_length=hop_length)
output_audio = np.sum(reconstructed_audio, axis=0)

filtered_audio = scipy.signal.medfilt(output_audio, kernel_size=3)

amplification_factor = 0.2
amplified_audio = filtered_audio * amplification_factor

librosa.output.write_wav('denoised_audio.wav', amplified_audio, target_sample_rate)

plot_signal_comparison(resampled_audio, denoised_signal, sample_rate)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(resampled_audio)
plt.title('Исходный сигнал')

plt.subplot(1, 2, 2)
plt.plot(noisy_signal)
plt.title('Шумный сигнал')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(resampled_audio)
plt.title('Исходный сигнал')

plt.subplot(1, 2, 2)
plt.plot(denoised_signal)
plt.title('Очищенный сигнал')

plt.tight_layout()
plt.show()
