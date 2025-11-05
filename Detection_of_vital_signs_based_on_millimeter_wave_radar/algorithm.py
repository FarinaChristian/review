from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
import pywt
from scipy.signal import welch

# https://doi.org/10.1038/s41598-025-09112-w

FS=25

def estimate_rates(resp, heart, fs, threshold=0.01):
    f_r, Pxx_r = welch(resp, fs)
    f_h, Pxx_h = welch(heart, fs)

    # Respirazione: 0.1 - 0.5 Hz
    mask_r = (f_r > 0.1) & (f_r < 0.5)
    resp_freq = f_r[mask_r][np.argmax(Pxx_r[mask_r])]
    resp_rate = resp_freq * 60
    
    # Battito cardiaco: 0.8 - 2 Hz
    mask_h = (f_h > 0.8) & (f_h < 2.0)
    peak = np.max(Pxx_h[mask_h])

    if peak < threshold:
        heart_rate = None   # come nel paper: battito non affidabile
    else:
        heart_freq = f_h[mask_h][np.argmax(Pxx_h[mask_h])]
        heart_rate = heart_freq * 60
    
    return resp_rate, heart_rate

# Kalman filtering
def akf(signal, Q=1e-5, R=0.01):
    x = 0
    P = 1
    filtered = []

    for z in signal:
        # Predict
        x = x
        P = P + Q
        
        # Update
        K = P / (P + R)
        x = x + K * (z - x)
        P = (1 - K) * P
        
        filtered.append(x)
    return np.array(filtered)

def extract_respiration_and_heartbeat(phase, wavelet='db5', max_level=4):
    # Determine how many levels are actually possible
    level = min(max_level, pywt.dwt_max_level(len(phase), pywt.Wavelet(wavelet).dec_len))
    
    # Valid decomposition
    coeffs = pywt.wavedec(phase, wavelet, level=level)
    
    # coeffs = [A_level, D_level, D_{level-1}, ..., D1]
    # Respiratory reconstruction (A-level only)
    resp_coeffs = [coeffs[0]] + [None] * level
    respiration = pywt.waverec(resp_coeffs, wavelet)

    # heartbeat reconstruction (solo D3 se esiste)
    # if level < 3, we adapt automatically
    heartbeat_coeffs = [None] + [None] * (level - 3) + [coeffs[-3]] + [None] * 2 if level >= 3 else None

    if level >= 3:
        heartbeat_coeffs = [None] + [None] * (level - 3) + [coeffs[-3]] + [None] * 2
        heartbeat = pywt.waverec(heartbeat_coeffs, wavelet)
    else:
        # fallback: get the highest detail available
        heartbeat = pywt.waverec([None] + coeffs[1:], wavelet)

    return respiration, heartbeat

def coherent_accumulation(support, num_pulses):
    support = np.asarray(support)
    N = len(support) // num_pulses
    support = support[:N * num_pulses].reshape(num_pulses, N)
    return np.mean(support, axis=0)

def estimate_breath_rate(data):
    # Step 1: FTT
    fft=np.fft.fft(data,axis=1)
    new_shape = (fft.shape[0] // Num_of_chirp_loops, Num_of_chirp_loops, fft.shape[1])
    fft = np.mean(fft.reshape(new_shape), axis=1) #average fft for each frame, every row is the fft of a frame 
    trans=abs(fft)# I get the magnitude
    trans = np.diff(trans, axis=0) #subtract consecutive frames to remove static objects

    #create a new array containing all the elements divided by 1000, used to remove useless peaks
    supporto = np.abs(trans) // 1000
    supporto[:, 0] = 0
    #print(" ".join(f"{i:.2f}" for i in supporto.mean(axis=0)))
    idx_max = np.argmax(supporto.mean(axis=0))
    print("Peak:",idx_max)

    # I get the signal in the index corresponding to the chest
    signal = fft[:, idx_max] 
    signal=coherent_accumulation(signal,32) # I don't Know how to set this parameter
    phase_unwrapped=np.unwrap(np.angle(signal))
    phase_diff=np.diff(phase_unwrapped) 

    filtered_signal_H,filtered_signal_B = extract_respiration_and_heartbeat(phase_diff)   
    resp=akf(filtered_signal_B)
    heart=akf(filtered_signal_H / (np.sqrt(np.mean(filtered_signal_H**2)) + 1e-8))
    
    return estimate_rates(resp,heart,FS)

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
    path="C:/Users\crist/Desktop/registrazioni/brAlta/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0])     
    print("Finished")
   
if __name__ == '__main__':
    main()
