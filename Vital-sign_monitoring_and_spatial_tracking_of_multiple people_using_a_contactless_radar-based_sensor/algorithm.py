from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
from scipy.signal import cwt, ricker
from scipy.ndimage import uniform_filter1d
import pywt

#10.1038/s41928-019-0258-6
FS=25

def detect_and_attenuate_artifacts(phase_signal, width_range=(1, 72)):
    widths = np.arange(width_range[0], width_range[1])
    cwt_coeffs = cwt(phase_signal, ricker, widths)
    artifact_energy = np.max(np.abs(cwt_coeffs), axis=0)
    
    threshold = np.mean(artifact_energy) + 3 * np.std(artifact_energy)
    mask = artifact_energy > threshold
    
    phase_filtered = phase_signal.copy()
    phase_filtered[mask] = uniform_filter1d(phase_filtered, size=5)[mask]
    return phase_filtered

def linear_demodulation_with_phase(I, Q):
    """
    Applica la demodulazione lineare e restituisce:
      y      : primo componente principale (segnale Doppler demodulato)
      phi_pca: fase dopo la demodulazione
      Y      : matrice dei componenti principali (2 x N)
      U      : matrice degli autovettori (2 x 2)
    """

    I = np.asarray(I)
    Q = np.asarray(Q)

    # 1) Rimozione della componente continua
    I_ac = I - np.mean(I)
    Q_ac = Q - np.mean(Q)

    # 2) Matrice MIQ (2 x N)
    MIQ = np.vstack((I_ac, Q_ac))

    # 3) Autovettori di MIQ * MIQ^T
    C = MIQ @ MIQ.T
    eigvals, U = np.linalg.eig(C)

    # Ordinamento per autovalori decrescenti
    idx = np.argsort(eigvals)[::-1]
    U = U[:, idx]

    # 4) Proiezione
    Y = U.T @ MIQ

    # Primo componente = nuovo segnale Doppler
    y = Y[0, :]

    # Fase dopo la demodulazione
    phi_pca = np.arctan2(Y[1, :], Y[0, :])

    return y, phi_pca, Y, U

def sliding_window(signal, window_samples, step_samples):
    #overlap=window_samples-step_samples
    for start in range(0, len(signal) - window_samples, step_samples):
        yield signal[start:start + window_samples]

def wavelet_separation(x, wavelet='db4', level=5, livelli_cuore=(3,4)):
    """
    Separa un segnale misto in componente respiratoria (bassa frequenza)
    e componente cardiaca (frequenza più alta) utilizzando wavelet decomposition.

    Parametri:
        x : array 1D
            Segnale misto.
        wavelet : str
            Tipo di wavelet da utilizzare.
        level : int
            Numero di livelli di decomposizione.
        livelli_cuore : tuple
            Indici dei dettagli da considerare per il segnale cardiaco.
            Esempio: (3,4) indica cD3 e cD4.

    Ritorna:
        resp : array 1D
            Componente respiratoria.
        cuore : array 1D
            Componente cardiaca.
    """

    # Decomposizione wavelet
    coeffs = pywt.wavedec(x, wavelet, level=level)

    # Costruzione componente respiratoria (solo approssimazione finale)
    c_resp = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    resp = pywt.waverec(c_resp, wavelet)

    # Costruzione componente cardiaca (solo i dettagli indicati)
    c_heart = []
    for i, c in enumerate(coeffs):
        # coeffs[0] = approssimazioni => metti sempre zero
        if i == 0:
            c_heart.append(np.zeros_like(c))
        else:
            # i=1 → cD(level), i=2 → cD(level-1), ecc.
            livello_detail = level - i + 1
            if livello_detail in livelli_cuore:
                c_heart.append(coeffs[i])
            else:
                c_heart.append(np.zeros_like(c))
    cuore = pywt.waverec(c_heart, wavelet)

    return resp, cuore

def calculateRate(filtrato):
    phaseFFT=np.fft.fft(filtrato)
    phaseFFT[0]=0
    frequencies = np.fft.fftfreq(len(phaseFFT), d=1/FS) 
    
    #I get the right frequency 
    value=frequencies[np.argmax(abs(phaseFFT[:len(phaseFFT)//2]))]
   
    return round(value*60,2)

def estimate_breath_rate(data):
    # Step 1: FTT
    trans=np.fft.fft(data,axis=1)
    new_shape = (trans.shape[0] // Num_of_chirp_loops, Num_of_chirp_loops, trans.shape[1])
    trans = np.mean(trans.reshape(new_shape), axis=1) #average fft for each frame, every row is the fft of a frame 
    mag=abs(trans)# I get the magnitude
    mag = np.diff(mag, axis=0) #subtract consecutive frames to remove static objects

    #create a new array containing all the elements divided by 1000, used to remove useless peaks
    supporto = np.abs(mag) // 1000
    supporto[:, 0] = 0
    #print(" ".join(f"{i:.2f}" for i in supporto.mean(axis=0)))

    print("Peak:",np.argmax(supporto.mean(axis=0)))

    # I get the phase in the index corresponding to the chest
    idx_max = np.argmax(supporto.mean(axis=0))
    bin = trans[:, idx_max]
    
    
    _,phase,_,_=linear_demodulation_with_phase(np.real(bin), np.imag(bin)) # I DON'T KNOW IF THE STEPS ARE IN THE CORRECT ORDER
    phase_clean=detect_and_attenuate_artifacts(phase)# the only thing I am sure of is that the phase is extracted correctly, if you unwrap it and 
                                                     # you perform the phase diff, you can extract the vital parameters

    window = int(FS * 60.0)      
    step   = int(FS * 60.0)      

    resp_segments = []
    heart_segments = []
    for seg in sliding_window(np.diff(np.unwrap(phase_clean)), window, step): # I am not supposed to unwrap the phase and perform phase diff
        resp, heart = wavelet_separation(seg)
        resp_segments.append(resp)
        heart_segments.append(heart) 

    return np.average([calculateRate(i) for i in heart_segments]),np.average([calculateRate(i) for i in resp_segments])

# it prints the final result
def printResult(adc_data,numFrames):
    acc,acc1,acc2,acc3=[],[],[],[]
    cont=0
    for frame in adc_data:
        accs = [acc, acc1, acc2, acc3]
        for i in range(4): # 4 è il numero di antenne
            accs[i] += list(frame[:, :, i]) #prendo le colonne identificate dal numero dell'antenna 1,2,3 e 4

        cont+=1
        if cont==numFrames:
            rateH,rateB= estimate_breath_rate(acc)
            print(f"ANTENNA 1 --> Peaks heart: {rateH} breath: {rateB}")
            rateH,rateB= estimate_breath_rate(acc1)
            print(f"ANTENNA 2 --> Peaks heart: {rateH} breath: {rateB}")
            rateH,rateB= estimate_breath_rate(acc2)
            print(f"ANTENNA 3 --> Peaks heart: {rateH} breath: {rateB}")
            rateH,rateB= estimate_breath_rate(acc3)
            print(f"ANTENNA 4 --> Peaks heart: {rateH} breath: {rateB}")
            cont=0
            acc.clear()
            acc1.clear()
            acc2.clear()
            acc3.clear()
            print("------------------")

def main():
    decoder = AWR1243()
    path="C:/Users/crist/Desktop/registrazioni/brAlta/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0])
    print("Finished")
   
if __name__ == '__main__':
    main()
