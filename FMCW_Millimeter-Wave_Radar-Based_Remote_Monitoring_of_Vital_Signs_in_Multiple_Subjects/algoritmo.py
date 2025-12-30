from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
from scipy.signal import ellip,filtfilt, find_peaks
import matplotlib.pyplot as plt
from utils.processing import heart_rate_fft2d, music
# 10.1109RCAR65431.2025.11139413
FS=25
 
def preparePhase(phaseMatrix,peak):
    #I get all the phases in the index of the moving object
    arrPhase=phaseMatrix[:,peak]

    phase_unwrapped=np.unwrap(arrPhase)
    phase_diff=np.diff(phase_unwrapped)# it contains the vital signs in the time domain

    b,a=ellip(4,1,40,[0.1/(FS*0.5),0.5/(FS*0.5)],btype='bandpass')
    filtered_signal_B = filtfilt(b,a,phase_diff) 
    b,a=ellip(4,1,40,[0.8/(FS*0.5),2/(FS*0.5)],btype='bandpass')
    filtered_signal_H = filtfilt(b,a,phase_diff) 

    return filtered_signal_B,filtered_signal_H

def estimate_breath_rate(data,antenna):
    # Step 1: FTT
    trans=np.fft.fft(data,axis=1)
    new_shape = (trans.shape[0] // Num_of_chirp_loops, Num_of_chirp_loops, trans.shape[1])
    trans = np.mean(trans.reshape(new_shape), axis=1) #average fft for each frame, every row is the fft of a frame 
    phase=np.angle(trans)# I get the phase
    trans=abs(trans)# I get the magnitude
    trans = np.diff(trans, axis=0) #subtract consecutive frames to remove static objects

    #create a new array containing all the elements divided by 1000, used to remove useless values
    supporto = np.abs(trans) // 1000
    supporto[:, 0] = 0
    arr=supporto.mean(axis=0)
    #print(" ".join(f"{i:.2f}" for i in supporto.mean(axis=0)))

    peaks, _ = find_peaks(arr)
    peaks = sorted(peaks, key=lambda i: arr[i], reverse=True)[:2]# I find the two highest peaks (according to the article)
    peaks.sort()
    print(antenna,"Peak:",*peaks)
    
    # I have only two peaks (two people), the article tries the algorithm with two people. We don't need a for cycle
    respiroP1,cuoreP1=preparePhase(phase,peaks[0])
    respiroP2,cuoreP2=preparePhase(phase,peaks[1])

    return respiroP1,cuoreP1,respiroP2,cuoreP2 # I return the BR and HR signals of two people

# it prints the final results considering all individuals
def printResult(adc_data,numFrames):
    acc,acc1,acc2,acc3=[],[],[],[]
    cont=0
    matriceBr1,matriceHr1,matriceBr2,matriceHr2=np.empty((4,numFrames-1)),np.empty((4,numFrames-1)),np.empty((4,numFrames-1)),np.empty((4,numFrames-1))
    for frame in adc_data:
        accs = [acc, acc1, acc2, acc3]
        for i in range(4): # 4 antennas
            accs[i] += list(frame[:, :, i]) #I get the columns identified by the index of the antenna
            
        cont+=1    
        if cont==numFrames:
            matriceBr1[0],matriceHr1[0],matriceBr2[0],matriceHr2[0] = estimate_breath_rate(acc,"ANTENNA 1")
            matriceBr1[1],matriceHr1[1],matriceBr2[1],matriceHr2[1] = estimate_breath_rate(acc1,"ANTENNA 2")
            matriceBr1[2],matriceHr1[2],matriceBr2[2],matriceHr2[2] = estimate_breath_rate(acc2,"ANTENNA 3")    
            matriceBr1[3],matriceHr1[3],matriceBr2[3],matriceHr2[3] = estimate_breath_rate(acc3,"ANTENNA 4")

            br1, _ , _= music(matriceBr1,FS)
            hr1=heart_rate_fft2d(matriceHr1)
            br2, _ , _=music(matriceBr2,FS)
            hr2=heart_rate_fft2d(matriceHr2)

            print("BR1",br1*60,"HR1",hr1)
            print("BR2",br2*60,"HR2",hr2)
            
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
