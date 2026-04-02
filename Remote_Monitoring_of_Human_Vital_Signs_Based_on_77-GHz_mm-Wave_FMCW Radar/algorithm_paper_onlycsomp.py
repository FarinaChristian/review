from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
from scipy.signal import ellip, filtfilt
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.fftpack import idct
import pywt
import matplotlib.pyplot as plt


#Frequenza di campionamento (dipende dal dataset)
FS = 20

def dwt(x, wavelet='db5', level=5, livelli_cuore=(3,4)):
    #Decomposizione wavelet
    coeffs = pywt.wavedec(x, wavelet, level=level)

    #Costruzione componente respiratoria
    c_resp = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    resp = pywt.waverec(c_resp, wavelet)

    #Costruzione componente cardiaca
    c_heart = []
    for i, c in enumerate(coeffs):
        if i == 0:
            c_heart.append(np.zeros_like(c))
        else:
            livello_detail = level - i + 1
            if livello_detail in livelli_cuore:
                c_heart.append(coeffs[i])
            else:
                c_heart.append(np.zeros_like(c))
    cuore = pywt.waverec(c_heart, wavelet)

    return resp, cuore

def cs_omp_respirazione(resp, compression_ratio=0.8, tol=1e-3, random_state=42):
    """
    MODIFICA CS-OMP: Rispetto all'uso di un numero fisso di coefficienti (che causa overfitting 
    sul rumore), qui usiamo una tolleranza dinamica (tol=1e-3). L'algoritmo estrae le componenti 
    necessarie e si ferma da solo quando l'errore residuo è trascurabile. 
    """
    resp = np.asarray(resp)
    N = resp.shape[0]
    rng = np.random.default_rng(random_state)
    M = int(compression_ratio * N)
    idx = np.sort(rng.choice(N, M, replace=False))

    Phi = np.zeros((M, N))
    Phi[np.arange(M), idx] = 1
    y = Phi @ resp

    Psi = idct(np.eye(N), norm='ortho')
    A = Phi @ Psi
    
    # La vera modifica: OMP guidato dall'errore residuo
    omp = OrthogonalMatchingPursuit(tol=tol)
    omp.fit(A, y)
    x_hat = omp.coef_

    risultato = Psi @ x_hat
    return risultato


def calculateRate(filtrato, f_min=0.0, f_max=3.0, breath_rate_bpm=None):
    phaseFFT = np.fft.fft(filtrato)
    phaseFFT[0] = 0
    frequencies = np.fft.fftfreq(len(phaseFFT), d=1/FS) 
    
    ampiezza = abs(phaseFFT[:len(phaseFFT)//2])
    freq_positive = frequencies[:len(frequencies)//2]
    
    #Mascheramento teorico rigoroso sulle bande del paper
    maschera = (freq_positive >= f_min) & (freq_positive <= f_max)
    freq_valide = freq_positive[maschera]
    amp_valide = ampiezza[maschera].copy()

    #Rimozione delle armoniche del respiro
    if breath_rate_bpm is not None and breath_rate_bpm > 0:
        f_br = breath_rate_bpm / 60.0  
        tolleranza = 0.05  
        for i in range(len(freq_valide)):
            f = freq_valide[i]
            ratio = f / f_br
            #Se la frequenza è un multiplo del respiro, azzera la sua energia
            if abs(ratio - round(ratio)) < (tolleranza / f_br):
                amp_valide[i] = 0  
                
    #Calcolo del picco nella zona teorica
    if len(amp_valide) > 0:
        value = freq_valide[np.argmax(amp_valide)]
    else:
        value = 0
   
    return round(value * 60, 2)


def estimate_breath_rate(data):
    #ESTRAZIONE DISTANZA
    trans = np.fft.fft(data, axis=1)
    new_shape = (trans.shape[0] // Num_of_chirp_loops, Num_of_chirp_loops, trans.shape[1])
    complex_frames = np.mean(trans.reshape(new_shape), axis=1) 
    
    mag_map = np.abs(complex_frames)
    dc_offset_mag = np.mean(mag_map, axis=1, keepdims=True)
    mag_clean = mag_map - dc_offset_mag
    
    support = np.abs(np.diff(mag_clean, axis=0)) // 1000
    support[:, 0] = 0 
    idx_max = np.argmax(support.mean(axis=0))

    print("Peak:",idx_max)

    #METODO DAC & CENTER TRACKING
    sig_complex = complex_frames[:, idx_max]
    sig_centered = sig_complex - np.mean(sig_complex)
    
    I = np.real(sig_centered)
    Q = np.imag(sig_centered)

    dI = np.diff(I)
    dQ = np.diff(Q)
    dac_signal = I[:-1] * dQ - Q[:-1] * dI
    
    if np.max(np.abs(dac_signal)) != 0:
        dac_signal = dac_signal / np.max(np.abs(dac_signal))

    #FILTRAGGIO TEORICO
    b_resp, a_resp = ellip(4, 1, 40, [0.1/(FS*0.5), 0.5/(FS*0.5)], btype='bandpass')
    filtered_resp = filtfilt(b_resp, a_resp, dac_signal) 

    #Filtro del cuore a 0.8 - 2.0 Hz
    b_heart, a_heart = ellip(6, 1, 50, [0.8/(FS*0.5), 2.0/(FS*0.5)], btype='bandpass')
    filtered_heart = filtfilt(b_heart, a_heart, dac_signal)

    #CALCOLO RATE (Correlazione tra DWT base e CS-OMP migliorato)
    # 1. Estrazione con Wavelet (base)
    resp_wave, _ = dwt(filtered_resp) 
    
    # 2. Estrazione con CS-OMP (migliorato con tolleranza dinamica)
    resp_omp = cs_omp_respirazione(filtered_resp)
    
    # 3. Cross-correlazione tra i due segnali come da paper
    resp_corr = np.correlate(resp_wave, resp_omp, mode='full')
    
    # 4. Calcolo del Rate finale sul segnale incrociato
    final_BR = calculateRate(resp_corr, f_min=0.1, f_max=0.5)

    #Calcolo Cuore nella sua banda, passando il BR per rimuovere le armoniche
    _, heart_wave = dwt(filtered_heart) 
    final_HR = calculateRate(heart_wave, f_min=0.8, f_max=2.0, breath_rate_bpm=final_BR)

    return final_HR, final_BR


def printResult(adc_data, numFrames):
    acc, acc1, acc2, acc3 = [], [], [], []
    cont = 0
    for frame in adc_data:
        accs = [acc, acc1, acc2, acc3]
        for i in range(4):
            accs[i] += list(frame[:, :, i])

        cont += 1
        if cont == numFrames:
            rateH, rateB = estimate_breath_rate(acc)
            rateH1, rateB1 = estimate_breath_rate(acc1)
            rateH2, rateB2 = estimate_breath_rate(acc2)
            rateH3, rateB3 = estimate_breath_rate(acc3)
            print(f" HR: {(rateH+rateH1+rateH2+rateH3)/4} BR: {(rateB+rateB1+rateB2+rateB3)/4}")
            cont = 0
            acc.clear(); acc1.clear(); acc2.clear(); acc3.clear()
            print("------------------")


def main():
    decoder = AWR1243()
    path = "C:/Users/crist/Desktop/registrazioni/christian-miriam-elena-gianni11/*" 
    print(f"Caricamento dati da: {path}...")
    adc_data = decoder.decode(path)
    
    if adc_data is not None:
        num_frames_disponibili = adc_data.shape[0]
        printResult(adc_data, num_frames_disponibili)
    else:
        print("Errore caricamento file.")

if __name__ == '__main__':
    main()



"""
VERSIONE TEORICA PAPER + MODIFICA CS-OMP:

1. MODIFICA CS-OMP (Respiro):
   Invece di ricostruire il segnale usando un numero fisso e rigido di coefficienti 
   (che porta l'algoritmo a fittare il rumore), è stata introdotta una 'tolleranza dinamica'
   all'errore (tol=1e-3). L'OMP estrae le componenti del segnale iterativamente finché 
   l'errore residuo scende sotto la soglia, garantendo una ricostruzione più accurata 
   della forma d'onda respiratoria.

2. COMPONENTI NON MODIFICATE (Cuore):
   - I filtri passa-banda IIR restano rigidamente ancorati alle soglie teoriche: 
     0.1-0.5 Hz (Respiro) e 0.8-2.0 Hz (Cuore).
   - L'estrazione dell'HR avviene tramite semplice ricerca del massimo assoluto (np.argmax) 
     nella banda teorica, senza l'ausilio di peak detection intelligente o soft masking.
   - Non c'è adattamento dinamico delle bande Wavelet.

Risultato: Migliore stima della frequenza fespiratoria (BR), ma persistenze di 
falsi positivi sulla frequenza cardiaca (HR) causati da rumore 1/f e errori di bordo banda.
"""