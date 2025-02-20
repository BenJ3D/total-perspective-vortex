import os
import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt

############################################################################
# 1) Fonctions d'affichage et d'extraction de features
############################################################################

def display_eeg_signals_and_spectra(data, times, title_suffix="",
                                    time_ylim=None, freq_xlim=(0, 100),
                                    fft_ylim=None):
    """
    Affiche pour chaque canal :
      - Le signal EEG en domaine temporel (même échelle si time_ylim).
      - Le spectre FFT (dans la plage freq_xlim) avec la même échelle en Y si fft_ylim est défini.

    Les figures sont organisées en grilles 8x8.
    On applique subplots_adjust pour contrôler manuellement les marges et l'espacement.
    """
    n_channels = data.shape[0]

    # Figure signaux temporels (8x8)
    fig_time, axs_time = plt.subplots(8, 8, figsize=(20, 20))
    axs_time = axs_time.flatten()
    # Ajuster les marges et l'espacement
    fig_time.subplots_adjust(left=0.07, right=0.986, top=0.967, bottom=0.06,
                             wspace=0.2, hspace=0.2)

    # Figure spectres de fréquence (8x8)
    fig_fft, axs_fft = plt.subplots(8, 8, figsize=(20, 20))
    axs_fft = axs_fft.flatten()
    # Ajuster les marges et l'espacement
    fig_fft.subplots_adjust(left=0.07, right=0.986, top=0.967, bottom=0.06,
                            wspace=0.2, hspace=0.2)

    # Vecteur de fréquences
    freqs = np.fft.rfftfreq(len(times), d=1/raw.info['sfreq'])

    for i in range(n_channels):
        # Affichage temporel
        axs_time[i].plot(times, data[i])
        axs_time[i].set_title(f"Canal {i} {title_suffix}")
        axs_time[i].set_xlabel("Temps (s)")
        axs_time[i].set_ylabel("Amplitude")
        axs_time[i].grid(True)
        if time_ylim is not None:
            axs_time[i].set_ylim(time_ylim)

        # Calcul FFT
        fft_spectrum = np.abs(np.fft.rfft(data[i]))
        axs_fft[i].plot(freqs, fft_spectrum)
        axs_fft[i].set_title(f"FFT Canal {i} {title_suffix}")
        axs_fft[i].set_xlabel("Fréquence (Hz)")
        axs_fft[i].set_ylabel("Amplitude")
        axs_fft[i].grid(True)
        axs_fft[i].set_xlim(freq_xlim)
        if fft_ylim is not None:
            axs_fft[i].set_ylim(fft_ylim)

    return fig_time, fig_fft


def extract_features_from_fft(data, times, frequency_bands, sfreq):
    """
    Extrait les puissances moyennes par bande de fréquences pour chaque canal.
    """
    n_channels = data.shape[0]
    freqs = np.fft.rfftfreq(len(times), d=1/sfreq)
    features_all = []
    for ch in range(n_channels):
        fft_values = np.fft.rfft(data[ch])
        power = np.abs(fft_values)**2
        features = []
        for band, (low, high) in frequency_bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            band_power = np.mean(power[idx]) if idx.size > 0 else 0
            features.append(band_power)
        features_all.append(features)
    return np.array(features_all)


############################################################################
# 2) Fonction utilitaire : calculer l'échelle FFT via un percentile (optionnel)
############################################################################

def get_fft_ylim(data_list, sfreq, percentile=99):
    """
    data_list : liste de matrices (n_channels, n_samples)
    Calcule la FFT de tous les canaux de toutes les matrices concaténées,
    puis renvoie un tuple (0, limit) où 'limit' est le percentile spécifié.
    """
    # Concatène les données sur l'axe 0 => shape (somme(n_channels), n_samples)
    all_data = np.concatenate(data_list, axis=0)
    # FFT sur l'axe 1 => shape (somme(n_channels), n_freq)
    all_fft = np.abs(np.fft.rfft(all_data, axis=1))
    limit = np.percentile(all_fft, percentile)
    return (0, limit)


############################################################################
# 3) Configuration et script principal
############################################################################

# Bandes de fréquences
frequency_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 30)
}

# Backend matplotlib
matplotlib.use("TkAgg")

# Chargement d'un fichier EDF
eeg_dir = os.getenv("EEG_DIR", "")
print("EEG DIR =", eeg_dir)
file = os.path.join(eeg_dir, "S001", "S001R01.edf")
raw = mne.io.read_raw_edf(file, preload=True)
print(raw.info)

# Données brutes
data_raw, times_raw = raw[:]

# Données filtrées (1-40 Hz)
raw_filtered = raw.copy()
raw_filtered.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
data_filt, times_filt = raw_filtered[:]

# Échelle commune pour l'axe Y temporel
global_time_min = min(data_raw.min(), data_filt.min())
global_time_max = max(data_raw.max(), data_filt.max())
common_time_ylim = (global_time_min, global_time_max)

# Échelle commune pour l'axe Y de la FFT via un percentile (par ex. 99e)
# Si tu veux utiliser la méthode percentile :
# common_fft_ylim = get_fft_ylim([data_raw, data_filt], sfreq=raw.info['sfreq'], percentile=99)
# Ou si tu veux fixer manuellement :
common_fft_ylim = (0, 0.03)

# Affichage Brute
fig_time_raw, fig_fft_raw = display_eeg_signals_and_spectra(
    data=data_raw,
    times=times_raw,
    title_suffix="- Brut",
    time_ylim=common_time_ylim,
    freq_xlim=(0, 80),
    fft_ylim=common_fft_ylim
)

# Affichage Filtrée
fig_time_filt, fig_fft_filt = display_eeg_signals_and_spectra(
    data=data_filt,
    times=times_filt,
    title_suffix="- Filtré",
    time_ylim=common_time_ylim,
    freq_xlim=(0, 80),
    fft_ylim=common_fft_ylim
)

# Extraction de features (optionnel)
features_all = extract_features_from_fft(data_filt, times_filt, frequency_bands, raw.info['sfreq'])
print("Features extraites (puissance moyenne par bande) pour chaque canal :")
print(features_all)

# Affichage final
plt.show()
