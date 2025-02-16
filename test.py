import os

import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt




def display_eeg__spectral_frequency():
    # Extraire les données brutes EEG du premier canal
    data, times = raw[:]
    # Calculer la FFT (Transformée de Fourier)
    freqs = np.fft.rfftfreq(len(times), d=1/raw.info['sfreq'])
    fft_spectrum = np.abs(np.fft.rfft(data[0]))  # Premier canal EEG

    # Afficher le spectre de fréquence
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, fft_spectrum)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Spectre de fréquence EEG")
    plt.xlim(0, 100)  # Afficher seulement jusqu'à 100 Hz
    plt.grid()
    # plt.show()


# pour fix bug ubuntu
# matplotlib.use("Qt5Agg")  # Change le backend
matplotlib.use("TkAgg")


eeg_dir = os.getenv("EEG_DIR", "")
print("EEG DIR =", eeg_dir)

file = eeg_dir + "/S001/S001R03.edf"
raw = mne.io.read_raw_edf(file, preload=True)
# print(f"Fréquence d'échantillonnage : {raw.info['sfreq']} Hz")
print(raw.info)

display_eeg__spectral_frequency()
plt.show()

raw.plot(duration=15, n_channels=20, scalings={'eeg': 400e-6})
# Appliquer un filtre passe-bande (1-40 Hz)
raw.filter(l_freq=8, h_freq=30, fir_design='firwin')

# Appliquer un notch filter à 60Hz (bruit électrique aux USA)
# raw.notch_filter(freqs=60)

print(raw.info)
# raw.plot(scalings='auto', duration=40)
# raw.plot(scalings={'eeg': 50e-6})
raw.plot(duration=15, n_channels=20, scalings={'eeg': 400e-6})

display_eeg__spectral_frequency()
plt.show()
