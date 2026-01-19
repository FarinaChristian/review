from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
import pywt
from scipy.signal import welch
import matplotlib.pyplot as plt

# https://doi.org/10.1038/s41598-025-09112-w

FS=20

def kalman_filter(signal, Q=1e-5, R=0.01):
    """
    Filtro di Kalman 1D adattivo per smussare un segnale.
    
    signal : np.array
        Segnale da filtrare.
    Q : float
        Varianza del processo (quanto il segnale può cambiare tra i campioni).
    R : float
        Varianza del rumore (quanto rumore è presente nel segnale).
    
    Restituisce:
    filtered_signal : np.array
        Segnale filtrato.
    """
    n = len(signal)
    x_est = np.zeros(n)  # stima del segnale
    P = np.zeros(n)      # errore di stima
    K = np.zeros(n)      # guadagno di Kalman

    # Inizializzazione
    x_est[0] = signal[0]
    P[0] = 1.0

    for k in range(1, n):
        # Predizione
        x_pred = x_est[k-1]
        P_pred = P[k-1] + Q

        # Aggiornamento
        K[k] = P_pred / (P_pred + R)
        x_est[k] = x_pred + K[k] * (signal[k] - x_pred)
        P[k] = (1 - K[k]) * P_pred

    return x_est

def estimate_breath_rate(data):
    # Step 1: FTT
    fft=np.fft.fft(data,axis=1)
    new_shape = (fft.shape[0] // Num_of_chirp_loops, Num_of_chirp_loops, fft.shape[1])
    fft = np.mean(fft.reshape(new_shape), axis=1) # average fft for each frame, every row is the fft of a frame 
    trans=abs(fft)# I get the magnitude
    trans = np.diff(trans, axis=0) # subtract consecutive frames to remove static objects

    # create a new array containing all the elements divided by 1000, used to remove useless peaks
    supporto = np.abs(trans) // 1000
    supporto[:, 0] = 0
    # print(" ".join(f"{i:.2f}" for i in supporto.mean(axis=0)))
    idx_max = np.argmax(supporto.mean(axis=0))
    print("Peak:",idx_max)

    # I get the signal in the index corresponding to the chest
    signal = fft[:, idx_max] 
    phase_unwrapped=np.unwrap(np.angle(signal))
    phase_diff=np.diff(phase_unwrapped) 
    #phase_diff=phase_diff-np.mean(phase_diff) # DC offset removal?, qui rimuovo il valore medio
    #phase_diff=phase_diff*10 # coherent accumulation? qui amplifico il segnale per renderlo più visibile
    #soluzione: uso detrend() per rimuovere il valore medio (DC offset) e la variazione del segnale
    from scipy.signal import detrend #questa riga e la successiva sono al posto delle due sopra con ?
    phase_diff = detrend(phase_diff)

    # 1. Decomposizione DWT multilevel
    coeffs = pywt.wavedec(phase_diff, 'db4', level=4)
    # coeffs = [A4, D4, D3, D2, D1]

    # 2. Segnale del respiro (basse frequenze) → approssimazione A4
    A4 = coeffs[0]
    coeffs_respiro = [A4] + [np.zeros_like(c) for c in coeffs[1:]]
    segnale_respiro = pywt.waverec(coeffs_respiro, 'db4')

    # 3. Segnale del cuore (frequenze medie) → dettaglio D3
    D3 = coeffs[-3]  # D3
    coeffs_cuore = [np.zeros_like(c) for c in coeffs]
    coeffs_cuore[-3] = D3
    segnale_cuore = pywt.waverec(coeffs_cuore, 'db4')
    
    #square root normalisation? Essa serve per ridurre la dinamica del segnale e a stabilizzarlo. Prepara il segnale per Kalman,
    #perché così riesce a filtrare meglio il segnale
    #segnale_respiro_norm = np.sign(segnale_respiro) * np.sqrt(np.abs(segnale_respiro)), l'articolo non definisce necessario usare la normalization per il respiro
    segnale_cuore_norm   = np.sign(segnale_cuore) * np.sqrt(np.abs(segnale_cuore)) 
    
    segnale_respiro_filtrato = kalman_filter(segnale_respiro)
    segnale_cuore_filtrato = kalman_filter(segnale_cuore_norm)

    # 1. PSD del segnale respiratorio
    f_respiro, Pxx_respiro = welch(segnale_respiro_filtrato, fs=FS, nperseg=1024)

    # 2. PSD del segnale cardiaco = SPECTRAL ANALYSIS
    f_cuore, Pxx_cuore = welch(segnale_cuore_filtrato, fs=FS, nperseg=1024) #qui uso una HR ESTIMATION, che è diversa da quella che otterrò alla fine
    #HR c'è l'ho in due occasioni: qui faccio una PSD preliminare per capire dove sta il cuore.
   
    br_band = (f_respiro >= 0.1) & (f_respiro <= 0.5) #valori presi dalla 1° tabella
    hr_band = (f_cuore   >= 0.8) & (f_cuore   <= 2.0)   

    #bande che impongono vincoli biologici, sempre quelli della 1° tabella
    Pxx_respiro_band = np.zeros_like(Pxx_respiro)
    Pxx_respiro_band[br_band] = Pxx_respiro[br_band]

    Pxx_cuore_band = np.zeros_like(Pxx_cuore)
    Pxx_cuore_band[hr_band] = Pxx_cuore[hr_band]

    BR_final = f_respiro[np.argmax(Pxx_respiro_band)] * 60
    HR_final = f_cuore[np.argmax(Pxx_cuore_band)] * 60

    return HR_final, BR_final

# it prints the final result
def printResult(adc_data,numFrames):
    acc,acc1,acc2,acc3=[],[],[],[]
    cont=0
    for frame in adc_data:
        accs = [acc, acc1, acc2, acc3]
        for i in range(4): # 4 antennas
            accs[i] += list(frame[:, :, i]) 

        cont+=1
        if cont==numFrames:
            rateH,rateB= estimate_breath_rate(acc)
            rateH1,rateB1= estimate_breath_rate(acc1)
            rateH2,rateB2= estimate_breath_rate(acc2)
            rateH3,rateB3= estimate_breath_rate(acc3)
            print(f" HR: {(rateH+rateH1+rateH2+rateH3)/4} BR: {(rateB+rateB1+rateB2+rateB3)/4}")
            cont=0
            acc.clear()
            acc1.clear()
            acc2.clear()
            acc3.clear()
            print("------------------")

def main():
    decoder = AWR1243()
    path = "C:/Users/crist/Desktop/registrazioni/muhamad-vasile-x-yosef/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0])     
    print("Finished")
   
if __name__ == '__main__':
    main()
