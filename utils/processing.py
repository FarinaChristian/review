from scipy.fftpack import fft, fftshift
import numpy as np
from constants.settings import Sample_per_chirp
from constants.settings import S
from scipy.signal import medfilt,butter,lfilter,find_peaks
from scipy.linalg import eigh

def first_fft(data, n_bins, hanning=True, shift=False):
    """Apply FFT across the ADC samples axis to get range information"""
    print("Computing fist FFT...")
    if hanning:
        data = data * np.hanning(data.shape[-2])[np.newaxis, np.newaxis, :, np.newaxis]

    range_fft = fft(data, n=n_bins, axis=2)

    if shift:
        range_fft = fftshift(range_fft, axes=2)

    return range_fft

def second_fft(data, n_bins, hanning=True, shift=False):
    """Apply FFT along the chirps axis to get Doppler information"""
    print("Computing second FFT...")
    if hanning:
        data = data * np.hanning(data.shape[-3])[np.newaxis, :, np.newaxis, np.newaxis]

    doppler_fft = fft(data, n=n_bins, axis=1)

    if shift:
        doppler_fft = fftshift(doppler_fft, axes=1)

    return doppler_fft

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

def dominant_freq_fft2(sig, fs=25, f_min=0.1, f_max=0.5, zp_factor=8):
    
    num_sensors, N = sig.shape

    # --- 1) Rimozione DC ---
    sig = sig - np.mean(sig, axis=1, keepdims=True)
    
    # --- 2) Finestratura ---
    window = np.hanning(N)
    sig = sig * window[None, :]

    # --- 3) FFT temporale con zero padding ---
    Nfft = zp_factor * N   # interpolazione spettrale
    fft_vals = np.fft.rfft(sig, n=Nfft, axis=1)
    
    # --- 4) Power spectrum multi-canale ---
    spectrum = np.sum(np.abs(fft_vals)**2, axis=0)

    # --- 5) Frequenze ---
    freqs = np.fft.rfftfreq(Nfft, 1/fs)

    # --- 6) Maschera fisiologica ---
    mask = (freqs >= f_min) & (freqs <= f_max)

    spectrum_masked = spectrum[mask]
    freqs_masked = freqs[mask]

    # --- 7) Picco grezzo ---
    k = np.argmax(spectrum_masked)

    # --- 8) Interpolazione parabolica del picco ---
    if 1 <= k < len(spectrum_masked)-1:
        y0, y1, y2 = spectrum_masked[k-1:k+2]
        d = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        freq_peak = freqs_masked[k] + d * (freqs_masked[1] - freqs_masked[0])
    else:
        freq_peak = freqs_masked[k]

    return freq_peak, freqs, spectrum


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
