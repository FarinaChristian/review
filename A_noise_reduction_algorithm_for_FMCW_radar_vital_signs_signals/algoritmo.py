import numpy as np
from vmdpy import VMD
import pywt
from scipy.signal import butter, filtfilt, find_peaks
from decoders.AWR1243 import AWR1243
from utils.processing import calculateRate

#Approssimazione della fuzzy entropy usata come fitness.
def fuzzy_entropy(x, m=2, r=None):
    x = np.asarray(x)

    if r is None:
        r = 0.2 * np.std(x)

    N = len(x)

    def phi(mm):
        patterns = np.array([x[i:i+mm] for i in range(N-mm+1)])
        D = np.max(np.abs(patterns[:, None] - patterns[None, :]),axis=2)
        mu = np.exp(-(D**2)/r)
        return np.sum(mu)/(len(patterns)**2)

    return np.log(phi(m)/phi(m+1))


def optimize_vmd(signal):

    best_score = np.inf
    best_K = None
    best_alpha = None

    for K in range(2, 10):

        for alpha in [500,1000,1500,2000,3000]:

            u, _, _ = VMD(signal,alpha=alpha,tau=0,K=K,DC=0,init=1,tol=1e-7)
            score = np.mean([fuzzy_entropy(imf) for imf in u])

            if score < best_score:
                best_score = score
                best_K = K
                best_alpha = alpha

    return best_K, best_alpha


def wavelet_denoise(imf,wavelet='sym5',level=3):

    coeffs = pywt.wavedec(imf,wavelet,level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    lam = sigma * np.sqrt(2*np.log(len(imf)))
    coeffs_denoised = [coeffs[0]]

    for c in coeffs[1:]:
        coeffs_denoised.append(new_threshold(c, lam))

    return pywt.waverec(coeffs_denoised,wavelet)[:len(imf)]


def new_threshold(w, lam, alpha=0.5, beta=2):
    absw = np.abs(w)
    out = np.zeros_like(w)
    mask = absw >= lam
    out[mask] = (np.sign(w[mask])* (absw[mask]- lam*np.tanh(((np.abs(lam-alpha))/absw[mask])**beta )))
    mask2 = ~mask
    out[mask2] = (np.sign(w[mask2])* ((absw[mask2]**(beta+1))/ (lam**beta))* ( 1- np.tanh( ((np.abs(lam-alpha))/absw[mask2])**beta)))
    return out

def ibka_vmd_wt(signal):

    K, alpha = optimize_vmd(signal)
    print(f"K={K}, alpha={alpha}")
    imfs, _, _ = VMD(signal,alpha=alpha,tau=0,K=K,DC=0,init=1,tol=1e-7)
    reconstructed = []

    for imf in imfs:

        cc = np.corrcoef(signal[:-1],imf)[0,1]

        if abs(cc) >= 0.8:
            # modalità "utile"
            reconstructed.append(imf)

        else:
            # modalità rumorosa
            reconstructed.append(wavelet_denoise(imf))

    denoised = np.sum(reconstructed,axis=0)
    return denoised


def bandpass(x, fs, low, high):
    b, a = butter(4,[low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def extract_vitals(signal, fs):
    respiration = bandpass(signal,fs,0.1,0.5)
    heartbeat = bandpass(signal,fs,0.9,2)
    return respiration, heartbeat

#--------------------------------------------------------------------------------

FS=25

def averageChirps(data):
    trans=np.fft.fft(data,axis=2)
    trans = np.mean(trans, axis=1)
    return trans

def findPeaksPeople(data):
    magnitude=abs(data)# I get the magnitude
    magnitude = np.diff(magnitude, axis=0) #subtract consecutive frames to remove static objects
    movingObjects = abs(magnitude)//1000
    movingObjects=np.mean(movingObjects,axis=(0,2))
    movingObjects[0] = 0
    return movingObjects

def preparePhase(data):
    phase=np.angle(data)# I get the phase
    phase_unwrapped=np.unwrap(phase)
    phase_diff=np.diff(phase_unwrapped)# it contains the vital signs in the time domain
    return phase_diff

# it prints the final results considering all individuals
def printResult(adc_data,numFrames,nomefile):
    structure3D= averageChirps(adc_data)       
    people=findPeaksPeople(structure3D)
    peaks1, _ = find_peaks(people)
    peaks, _ = find_peaks(people, height=people[peaks1].std()-people[peaks1].mean())
    print("Peak:",*peaks) 

    for i in range(0,adc_data.shape[0],numFrames):
        #preparazione della matrice da scrivere sul file csv
        matriceDaScrivere = np.empty((len(peaks),3), dtype=object)
        matriceDaScrivere[0:len(peaks),0]=peaks
        matriceDaScrivere[0,0]=nomefile+"-"+str(matriceDaScrivere[0,0])
        for j,p in enumerate(peaks):
            completo=preparePhase(structure3D[i:i+numFrames,p,0]) # first antenna
            completo1=preparePhase(structure3D[i:i+numFrames,p,1]) # second antenna
            completo2=preparePhase(structure3D[i:i+numFrames,p,2]) # third antenna
            completo3=preparePhase(structure3D[i:i+numFrames,p,3]) # fourth antenna

            denoised = ibka_vmd_wt((completo+completo1+completo2+completo3)/4)

            resp, heart = extract_vitals(denoised,FS)

            BR=calculateRate(resp, "BR", FS, grafico=False)
            HR=calculateRate(heart, "HR", FS, grafico=False)
    
            print(f"HR: {HR}, BR: {BR}") 
     
     
        print("------------------")

def main():
    decoder = AWR1243()
    path="C:/Users/crist/Desktop/registrazioni/christian/*"
    adc_data = decoder.decode(path)
    print(adc_data.shape)
    printResult(adc_data,adc_data.shape[0],"")
    print("Finished")
   
if __name__ == '__main__':
    main()