from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
from utils.processing import music_respiration, remove_baseline_drift,band_pass_filter_1d
import padasip as pa

# https://doi.org/10.1038/s41598-024-77683-1
FS=25

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
    
    return picchiH*60,calculateRate(filtered_signal_B)

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
    path="C:/Users/crist/Desktop/registrazioni/brBassa/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0])
            
    print("Finished")
   
if __name__ == '__main__':
    main()
