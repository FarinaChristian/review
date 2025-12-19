from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
from scipy.signal import ellip,filtfilt
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.fftpack import idct
import pywt

'''
This code contains an algorithm for extracting vital signs from some files recorded with an FMCW radar. 
The work consists of completing some missing parts. (You can find the explanation of the missing parts in the article)
'''

FS=25

def dwt(x, wavelet='db5', level=5, livelli_cuore=(3,4)):

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

def cs_omp_respirazione(resp,compression_ratio=0.3,n_nonzero_coefs=10,random_state=None):
    resp = np.asarray(resp)
    N = resp.shape[0]
    # -------------------------
    # Sottocampionamento
    # -------------------------
    rng = np.random.default_rng(random_state)
    M = int(compression_ratio * N)
    idx = np.sort(rng.choice(N, M, replace=False))

    Phi = np.zeros((M, N))
    Phi[np.arange(M), idx] = 1
    y = Phi @ resp

    # -------------------------
    # Dizionario sparso (DCT)
    # -------------------------
    Psi = idct(np.eye(N), norm='ortho')
    A = Phi @ Psi
    # -------------------------
    # OMP
    # -------------------------
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)

    omp.fit(A, y)
    x_hat = omp.coef_

    # -------------------------
    # Ricostruzione
    # -------------------------
    risultato = Psi @ x_hat

    return risultato # signal in the time domain

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
    trans=abs(trans)# I get the magnitude, it is the Range-Time Map, each row is a frame, each frame is a point in the time (Y-axs=time, X-axis=space)

    #-----------------------------------------------------
    #HERE YOU NEED TO APPLY THE DC OFFSET CORRECTION 
    #-----------------------------------------------------

    trans = np.diff(trans, axis=0) #subtract consecutive frames to remove static objects

    #create a new array containing all the elements divided by 1000, used to remove useless peaks
    support = np.abs(trans) // 1000
    support[:, 0] = 0
    idx_max = np.argmax(support.mean(axis=0))
    #print(" ".join(f"{i:.2f}" for i in support.mean(axis=0)))

    print("Peak:",idx_max)

    # I get the phase in the index corresponding to the chest
    arrPhase = phase[:, idx_max] 

    phase_unwrapped=np.unwrap(arrPhase)
    phase_diff=np.diff(phase_unwrapped) 

    b,a=ellip(4,1,40,[0.1/(FS*0.5),0.5/(FS*0.5)],btype='bandpass')
    filtered_signal_B = filtfilt(b,a,phase_diff) 
    b,a=ellip(4,1,40,[0.8/(FS*0.5),2/(FS*0.5)],btype='bandpass')
    filtered_signal_H = filtfilt(b,a,phase_diff)   

    resp, HR=dwt(phase_diff)
    resp2= cs_omp_respirazione(filtered_signal_B)
    BR = np.correlate(resp, resp2, mode='full')
    
    # THIS IS MY OLD ALGORITHM, YOU NEED TO RETURN THE RESULTS OBTAINED BY THE AUTOCORRELATION METHOD DESCRIBED IN THE ARTICLE
    return calculateRate(HR),calculateRate(BR) 

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
    path="C:/Users/crist/Desktop/registrazioni/christian4/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0])
    print("Finished")
   
if __name__ == '__main__':
    main()
