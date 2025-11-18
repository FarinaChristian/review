from constants.settings import Num_of_chirp_loops
from decoders.AWR1243 import AWR1243
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from utils.processing import band_pass_filter_1d

# https://doi.org/10.1145/3478090

FS=25

#it removes close peaks, it choses only the highest one
def rimuovi_vicini(peaks, values, distanza_minima=1):
    # Ordina i picchi in base ai valori decrescenti, per tenere quello più alto
    sorted_indices = np.argsort(values)[::-1]  # Indici ordinati dal valore più alto al più basso
    peaks = peaks[sorted_indices]
    values = values[sorted_indices]

    # Lista per conservare i picchi selezionati
    picchi_finali = []
    valori_finali = []

    for i in range(len(peaks)):
        # Controlla se il picco è vicino a uno già accettato
        if all(abs(peaks[i] - p) > distanza_minima for p in picchi_finali):
            picchi_finali.append(peaks[i])
            valori_finali.append(values[i])

    # Riordina i risultati secondo gli indici originali
    sorted_indices = np.argsort(picchi_finali)
    return np.array(picchi_finali)[sorted_indices], np.array(valori_finali)[sorted_indices]

# it finds peaks considering the distance: the farther the person is, the lower the peak
def find_peaks_with_varying_thresholds(data, thresholds, intervals):
    peaks = []
    peak_values = []

    for (start, end), min_height in zip(intervals, thresholds):
        # Trova i picchi nella finestra specificata
        local_peaks, properties = scipy.signal.find_peaks(data[start:end], height=min_height)
        
        # Aggiusta gli indici per l'array completo
        adjusted_peaks = local_peaks + start
        peaks.extend(adjusted_peaks)
        peak_values.extend(data[adjusted_peaks])  # Ottiene i valori dei picchi

    return np.array(peaks), np.array(peak_values)

def calculateRate(filtrato):
    phaseFFT=np.fft.fft(filtrato)
    phaseFFT[0]=0
    frequencies = np.fft.fftfreq(len(phaseFFT), d=1/FS) 
   
    '''fig2, (ax2) = plt.subplots(1, 1, figsize=(10, 10))
    ax2.plot(abs(phaseFFT[:len(phaseFFT)//2]))
    fig2.canvas.draw()
    plt.xlabel("Breathing rates",fontsize=14)
    plt.ylabel("Magnitude",fontsize=14)
    plt.show()'''
    
    #I get the right frequency 
    value=frequencies[np.argmax(abs(phaseFFT[:len(phaseFFT)//2]))]

    return round(value*60,2)
    
def preparePhase(phaseMatrix,peak):
    #I get all the phases in the index of the moving object
    arrPhase=phaseMatrix[:,peak]

    phase_unwrapped=np.unwrap(arrPhase)
    phase_diff=np.diff(phase_unwrapped)# it contains the vital signs in the time domain

    '''frequencies = np.fft.fftfreq(len(phase_diff), d=1/FS) 
    phaseFFT=np.fft.fft(phase_diff)
    fig2, (ax2) = plt.subplots(1, 1, figsize=(10, 10))
    ax2.plot(frequencies[:len(phaseFFT)//2], abs(phaseFFT[:len(phaseFFT)//2]))
    fig2.canvas.draw()
    plt.show()'''

    filtered_signal_H = band_pass_filter_1d(phase_diff,FS, 0.9, 1.67) #heart
    filtered_signal_B = band_pass_filter_1d(phase_diff,FS, 0.1, 0.50) #breath
    
    return calculateRate(filtered_signal_H),calculateRate(filtered_signal_B)

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
    #print(" ".join(f"{i:.2f}" for i in supporto.mean(axis=0)))

    '''fig2, (ax2) = plt.subplots(1, 1, figsize=(10, 10))
    ax2.plot([0,12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0, 112.5, 125.0,137.5, 150.0, 162.5, 175.0, 187.5, 200.0, 212.5, 225.0, 237.5, 250.0],supporto.mean(axis=0)[:21])
    fig2.canvas.draw()
    plt.xlabel("Distance (cm)",fontsize=14)
    plt.ylabel("Magnitude",fontsize=14)
    plt.show()'''

    # la soglia 1 non serve a niente, è un rimasuglio della versione con le prime impostazioni, la soglia 2 mi dice quanto "potente" è
    # il petto che si muove, serve per trovare effettivamente le persone presenti dopo la prima trasformata.
    soglia1=0.5
    soglia2=0.5
    peaks,values=find_peaks_with_varying_thresholds(supporto.mean(axis=0), [soglia1,soglia2], [(0,11),(9,len(supporto.mean(axis=0)))])
    peaks,values=rimuovi_vicini(peaks,values)
    print(antenna,"Peak:",*peaks)
    
    heartBreath=[] #it contains the tuples with the heart beats and the breath rates
    for p in peaks:
        heartBreath.append(preparePhase(phase,p))

    return heartBreath # the array is as long as the number of people

# it prints the final results considering all individuals
def printResult(adc_data,numFrames):
    acc,acc1,acc2,acc3=[],[],[],[]
    cont=0
    for frame in adc_data:
        accs = [acc, acc1, acc2, acc3]
        for i in range(4): # 4 è il numero di antenne
            accs[i] += list(frame[:, :, i]) #prendo le colonne identificate dal numero dell'antenna 1,2,3 e 4
            
        cont+=1    
        if cont==numFrames:
            heartBreath = estimate_breath_rate(acc,"ANTENNA 1")
            for num,i in enumerate(heartBreath):
                print(f"Person {num} --> Heartbeat: {i[0]}, Breath rate: {i[1]}")

            heartBreath = estimate_breath_rate(acc1,"ANTENNA 2")
            for num,i in enumerate(heartBreath):
                print(f"Person {num} --> Heartbeat: {i[0]}, Breath rate: {i[1]}")

            heartBreath = estimate_breath_rate(acc2,"ANTENNA 3")
            for num,i in enumerate(heartBreath):
                print(f"Person {num} --> Heartbeat: {i[0]}, Breath rate: {i[1]}")

            heartBreath = estimate_breath_rate(acc3,"ANTENNA 4")
            for num,i in enumerate(heartBreath):
                print(f"Person {num} --> Heartbeat: {i[0]}, Breath rate: {i[1]}")
            cont=0
            acc.clear()
            acc1.clear()
            acc2.clear()
            acc3.clear()
            print("------------------")


def main():
    decoder = AWR1243()
    path="C:/Users/crist/Desktop/registrazioni/ailati/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0])       
    print("Finished")
   
if __name__ == '__main__':
    main()
