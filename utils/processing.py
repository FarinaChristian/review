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
    freq_peak = freqs[idx_peak]
    
    return freq_peak, freqs, spectrum

def dominant_freq_fft2(sig, fs=25, f_min=0.1, f_max=0.5):
    # --- Step 1: rimuovi DC su ogni canale ---
    sig_centered = sig - np.mean(sig, axis=1, keepdims=True)
    
    # --- Step 2: FFT 2D ---
    fft2_vals = np.fft.fft2(sig_centered)
    mag = np.abs(fft2_vals)
    
    # --- Step 3: somma energia lungo i canali (righe) ---
    spectrum = mag.sum(axis=0)  # spettro 1D lungo il tempo
    
    # --- Step 4: frequenze associate ---
    N = sig.shape[1]
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
    spectrum = spectrum[:N//2]
    
    # --- Step 5: limita al range fisiologico ---
    mask = (freqs >= f_min) & (freqs <= f_max)
    idx_peak = np.argmax(spectrum[mask])
    freq_peak = freqs[mask][idx_peak]
    
    return freq_peak, freqs, spectrum


def music(sig_matrix, fs=25, M=150, num_sources=1, n_freqs=1000):
    num_sensori, N = sig_matrix.shape

    # --- Step 1: Costruzione Hankel per ciascun sensore ---
    X_all = []
    for sig in sig_matrix:
        # Hankel per questo segnale
        X = np.array([sig[i:N-M+i+1] for i in range(M)])
        X_all.append(X)
    # Concateno tutte le righe insieme
    X_all = np.vstack(X_all)  # dimensione totale: num_sensori*M × (N-M+1)

    # --- Step 2: Covarianza e sottospazio del rumore ---
    R = X_all @ X_all.conj().T / X_all.shape[1]
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    E_n = eigvecs[:, num_sources:]  # noise subspace

    # --- Step 3: Spettro MUSIC ---
    freqs = np.linspace(0, fs/2, n_freqs)
    M_total = X_all.shape[0]  # numero totale di righe
    A = np.exp(-1j * 2 * np.pi * np.outer(np.arange(M_total), freqs/fs))
    v = E_n.conj().T @ A
    spectrum = 1 / np.sum(np.abs(v)**2, axis=0)

    # --- Step 4: Frequenza dominante ---
    idx_peak = np.argmax(spectrum)
    freq_peak = freqs[idx_peak]

    return freq_peak, freqs, spectrum