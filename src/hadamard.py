import numpy as np

def hadamard_transform(signal):
    N = len(signal)
    H = hadamard_matrix(N)

    transformed_signal = np.matmul(H, signal)

    return transformed_signal

def inverse_hadamard_transform(transformed_signal):
    N = len(transformed_signal)
    H = hadamard_matrix(N)

    inverse_transformed_signal = np.matmul(H, transformed_signal) / N

    return inverse_transformed_signal

def hadamard_matrix(N):
    if N == 1:
        return np.array([[1]])

    H_prev = hadamard_matrix(N // 2)
    H_top = np.concatenate((H_prev, H_prev), axis=1)
    H_bottom = np.concatenate((H_prev, -H_prev), axis=1)
    H = np.concatenate((H_top, H_bottom), axis=0)

    return H

def add_noise(signal, noise_std):
    noise = np.random.normal(0, noise_std, size=len(signal))

    noisy_signal = signal + noise

    return noisy_signal

def denoise_signal(noisy_signal, noise_std):
    transformed_signal = hadamard_transform(noisy_signal)

    amplitude = np.abs(transformed_signal)

    threshold = noise_std * np.sqrt(np.log2(len(amplitude)))
    amplitude[amplitude <= threshold] = 0

    denoised_signal = inverse_hadamard_transform(transformed_signal)

    return denoised_signal
