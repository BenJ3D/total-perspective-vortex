import os

import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt


def display_all_eeg_signals_and_spectra():
    # Créer une copie du signal brut pour ne pas modifier l'original
    raw_filtered = raw.copy()
    # Appliquer un filtre passe-bande 1-40 Hz (en utilisant la méthode FIR)
    raw_filtered.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

    # Extraire les données brutes EEG filtrées
    data, times = raw_filtered[:]
    print("data shape =", data.shape)  # Affiche la forme : (64, 20000)
    n_channels = data.shape[0]

    # Créer une figure pour afficher les signaux EEG en domaine temporel pour tous les canaux
    # On crée une grille de 8 lignes x 8 colonnes (pour 64 canaux)
    fig_time, axs_time = plt.subplots(8, 8, figsize=(20, 20))
    axs_time = axs_time.flatten()

    # Créer une figure pour afficher les spectres de fréquence pour tous les canaux
    fig_fft, axs_fft = plt.subplots(8, 8, figsize=(20, 20))
    axs_fft = axs_fft.flatten()

    # Calculer le vecteur de fréquences commun à tous les canaux
    freqs = np.fft.rfftfreq(len(times), d=1 / raw.info['sfreq'])

    # Boucle sur tous les canaux
    for i in range(n_channels):
        # Afficher le signal EEG en domaine temporel pour le canal i
        axs_time[i].plot(times, data[i])
        axs_time[i].set_title(f"Canal {i}")
        axs_time[i].set_xlabel("Temps (s)")
        axs_time[i].set_ylabel("Amplitude")
        axs_time[i].grid(True)

        # Calculer la FFT pour le canal i
        fft_spectrum = np.abs(np.fft.rfft(data[i]))
        # Afficher le spectre de fréquence pour le canal i
        axs_fft[i].plot(freqs, fft_spectrum)
        axs_fft[i].set_title(f"FFT Canal {i}")
        axs_fft[i].set_xlabel("Fréquence (Hz)")
        axs_fft[i].set_ylabel("Amplitude")
        axs_fft[i].grid(True)
        axs_fft[i].set_xlim(0, 40)  # On limite l'affichage à 40 Hz (optionnel)

    plt.tight_layout()
    # plt.show()  # Affichage différé à la fin du script


# pour fix bug ubuntu
# matplotlib.use("Qt5Agg")  # Change le backend
matplotlib.use("TkAgg")

eeg_dir = os.getenv("EEG_DIR", "")
print("EEG DIR =", eeg_dir)

file = eeg_dir + "/S023/S023R10.edf"
raw = mne.io.read_raw_edf(file, preload=True)
# print(f"Fréquence d'échantillonnage : {raw.info['sfreq']} Hz")
print(raw.info)
# print("TEST : ")
# print(raw._data)

# Afficher le signal EEG filtré 1-40 Hz et son spectre pour chaque canal
display_all_eeg_signals_and_spectra()
plt.show()

#
# raw.plot(duration=15, n_channels=20, scalings={'eeg': 400e-6})
# # Appliquer un filtre passe-bande (1-40 Hz)
# # raw.filter(l_freq=8, h_freq=70, fir_design='firwin')
#
# # Appliquer un notch filter à 60Hz (bruit électrique aux USA)
# # raw.notch_filter(freqs=60)
#
# print(raw.info)
# # raw.plot(scalings='auto', duration=40)
# # raw.plot(scalings={'eeg': 50e-6})
# raw.plot(duration=15, n_channels=20, scalings={'eeg': 400e-6})
#
# # Réafficher après filtrage
# # display_all_eeg_signals_and_spectra()
# # plt.show()
