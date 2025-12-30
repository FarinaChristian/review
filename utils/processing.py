from scipy.fftpack import fft, fftshift
import numpy as np
from constants.settings import Sample_per_chirp
from constants.settings import S
from scipy.signal import medfilt,butter,lfilter,find_peaks
from scipy.linalg import eigh

def band_pass_filter_1d(data: np.ndarray, sampling_frequency: float, low_cut: float, high_cut: float) -> np.ndarray:
    """ Applica un filtro passa-banda Butterworth a un array monodimensionale complesso. """
    # Progetta un filtro Butterworth passa-banda
    b, a = butter(N=4, Wn=[low_cut, high_cut], btype='band', fs=sampling_frequency)

    # Applica il filtro separatamente a parte reale e immaginaria
    filtered_real = lfilter(b, a, data.real)
    filtered_imag = lfilter(b, a, data.imag)

    # Ricostruisce il segnale complesso
    return filtered_real + 1j * filtered_imag

def remove_baseline_drift(signal, window_size=101):
    # Controllo che la finestra sia dispari
    if window_size % 2 == 0:
        raise ValueError("La finestra deve essere un numero dispari.")

    if np.iscomplexobj(signal):
        # Filtra parte reale e immaginaria separatamente
        b_real = medfilt(signal.real, kernel_size=window_size)
        b_imag = medfilt(signal.imag, kernel_size=window_size)
        b_t = b_real + 1j * b_imag
    else:
        # Segnale reale
        b_t = medfilt(signal, kernel_size=window_size)

    # Segnale corretto
    return signal - b_t

def music_respiration(sig, fs=25, M=150, num_sources=1, n_freqs=1000):
    """Stima la frequenza respiratoria dominante di un segnale 1D usando MUSIC."""
    # --- Step 1: Matrice Hankel ---
    N = len(sig)
    X = np.array([sig[i:N-M+i+1] for i in range(M)])
    
    # --- Step 2: Covarianza e subspace di rumore ---
    R = X @ X.conj().T / X.shape[1]
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    E_n = eigvecs[:, num_sources:]  # noise subspace
    
    # --- Step 3: Steering vector temporale ---
    freqs = np.linspace(0, fs/2, n_freqs)
    A = np.exp(-1j * 2 * np.pi * np.outer(np.arange(M), freqs/fs))
    
    # --- Step 4: Spettro MUSIC ---
    v = E_n.conj().T @ A
    spectrum = 1 / np.sum(np.abs(v)**2, axis=0)
    
    # --- Step 5: Frequenza dominante ---
    idx_peak = np.argmax(spectrum)
    freq_peak = freqs[idx_peak]*60
    
    return freq_peak, freqs, spectrum

def heart_rate_fft2d(signals, Fs=25):
    # FFT 2D
    F = np.fft.fftshift(np.fft.fft2(signals))
    mag = np.abs(F)

    # asse frequenze (tempo)
    N = signals.shape[1]
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))

    # rimuovi DC
    mag[:, N//2] = 0

    # picco globale
    _, ix = np.unravel_index(np.argmax(mag), mag.shape)

    f_hr = fx[ix]
    bpm = abs(f_hr) * 60

    return bpm

def music(sig_matrix, fs=25, M=150, num_sources=1, n_freqs=10000):

    num_sensors, N = sig_matrix.shape
    L = N - M + 1

    # --- 1) Hankel per ogni sensore ---
    covariances = []

    for sig in sig_matrix:
        X = np.zeros((M, L), dtype=complex)
        for i in range(M):
            X[i] = sig[i:i+L]

        # Covarianza del singolo sensore
        R = X @ X.conj().T / L
        covariances.append(R)

    # --- 2) Media delle covarianze ---
    R = sum(covariances) / num_sensors

    # Stabilizzazione numerica
    R += 1e-10 * np.eye(M)

    # --- 3) Decomposizione autovalori ---
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    E_n = eigvecs[:, num_sources:]   # noise subspace

    # --- 4) MUSIC spectrum ---
    freqs = np.linspace(0, fs/2, n_freqs)

    m = np.arange(M)
    A = np.exp(-1j * 2*np.pi * m[:,None] * freqs / fs)

    proj = E_n.conj().T @ A
    spectrum = 1 / np.sum(np.abs(proj)**2, axis=0)

    # Normalizzazione
    spectrum /= spectrum.max()

    # --- 5) Picco ---
    idx_peak = np.argmax(spectrum)
    freq_peak = freqs[idx_peak]

    return freq_peak, freqs, spectrum
