from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
import padasip as pa
from scipy.signal import medfilt,butter,lfilter

# https://doi.org/10.1038/s41598-024-77683-1
FS=25

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
    
    N = len(sig)
    X = np.array([sig[i:N-M+i+1] for i in range(M)])

    R = X @ X.conj().T / X.shape[1]
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    E_n = eigvecs[:, num_sources:]  
    
    freqs = np.linspace(0, fs/2, n_freqs)
    A = np.exp(-1j * 2 * np.pi * np.outer(np.arange(M), freqs/fs))
    v = E_n.conj().T @ A 

    spectrum = 1 / np.sum(np.abs(v)**2, axis=0)
    idx_peak = np.argmax(spectrum)
    freq_peak = freqs[idx_peak]*60
    return freq_peak, freqs, spectrum

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
    phase=np.angle(trans)# I get the phase
    trans=abs(trans)# I get the magnitude
    trans = np.diff(trans, axis=0) #subtract consecutive frames to remove static objects

    #create a new array containing all the elements divided by 1000, used to remove useless peaks
    supporto = np.abs(trans) // 1000
    supporto[:, 0] = 0
    #print(" ".join(f"{i:.2f}" for i in supporto.mean(axis=0)))

    print("Peak:",np.argmax(supporto.mean(axis=0)))

    peak = np.argmax(supporto.mean(axis=0))
    arrPhase = phase[:, peak]  

    phase_unwrapped=np.unwrap(arrPhase) 
    differenza=np.diff(phase_unwrapped)
    filtered_signal_B = band_pass_filter_1d(differenza,FS, 0.1, 0.5)

    # this is the new part described in the article
    U=remove_baseline_drift(phase_unwrapped,35)
    filt=pa.filters.FilterRLS(4,mu=0.99,w="random")
    X=pa.input_from_history(filtered_signal_B,4)
    y,e,w=filt.run(U[4:],X)# e is U-y
    phase_diff = np.diff(e) 
    
    picchiH,_, _ = music_respiration(phase_diff)
    
    return picchiH,calculateRate(filtered_signal_B)

# it prints the final result
def printResult(adc_data,numFrames):
    acc,acc1,acc2,acc3=[],[],[],[]
    cont=0
    for frame in adc_data:
        accs = [acc, acc1, acc2, acc3]
        for i in range(4): # 4 antennas
            accs[i] += list(frame[:, :, i]) #I get the columns identified by the index of the antenna

        cont+=1
        if cont==numFrames:
            rateH,rateB= estimate_breath_rate(acc)
            print(f"ANTENNA 1 --> heart: {rateH} breath: {rateB}")
            rateH,rateB= estimate_breath_rate(acc1)
            print(f"ANTENNA 2 --> heart: {rateH} breath: {rateB}")
            rateH,rateB= estimate_breath_rate(acc2)
            print(f"ANTENNA 3 --> heart: {rateH} breath: {rateB}")
            rateH,rateB= estimate_breath_rate(acc3)
            print(f"ANTENNA 4 --> heart: {rateH} breath: {rateB}")
            cont=0
            acc.clear()
            acc1.clear()
            acc2.clear()
            acc3.clear()
            print("------------------")

def main():
    decoder = AWR1243()
    path="C:/Users/crist/Desktop/registrazioni/christian5/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0])
            
    print("Finished")
   
if __name__ == '__main__':
    main()
