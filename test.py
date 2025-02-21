import os
import re
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import warnings

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score


############################################################################
# 1) Fonctions d'affichage et d'extraction de features
############################################################################

def display_eeg_signals_and_spectra(data, times, sfreq, title_suffix="",
                                    time_ylim=None, freq_xlim=(0, 100),
                                    fft_ylim=None, grid_shape=None,
                                    custom_margins=None):
    """
    Affiche pour chaque canal :
      - Le signal EEG en domaine temporel.
      - Le spectre FFT (dans la plage freq_xlim).
    """
    n_channels = data.shape[0]
    if grid_shape is None:
        n_rows = int(np.ceil(np.sqrt(n_channels)))
        n_cols = int(np.ceil(n_channels / n_rows))
    else:
        n_rows, n_cols = grid_shape

    # Figure des signaux temporels
    fig_time, axs_time = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_channels == 1:
        axs_time = [axs_time]
    else:
        axs_time = np.array(axs_time).flatten()

    # Figure des spectres FFT
    fig_fft, axs_fft = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_channels == 1:
        axs_fft = [axs_fft]
    else:
        axs_fft = np.array(axs_fft).flatten()

    # Ajustement des marges
    if custom_margins is not None:
        fig_time.subplots_adjust(**custom_margins)
        fig_fft.subplots_adjust(**custom_margins)
    else:
        fig_time.subplots_adjust(left=0.07, right=0.986, top=0.967, bottom=0.06,
                                 wspace=0.2, hspace=0.2)
        fig_fft.subplots_adjust(left=0.07, right=0.986, top=0.967, bottom=0.06,
                                wspace=0.2, hspace=0.2)

    # Calcul du vecteur de fréquences
    freqs = np.fft.rfftfreq(len(times), d=1 / sfreq)

    for i in range(n_channels):
        axs_time[i].plot(times, data[i])
        axs_time[i].set_title(f"Canal {i} {title_suffix}")
        axs_time[i].set_xlabel("Temps (s)")
        axs_time[i].set_ylabel("Amplitude")
        axs_time[i].grid(True)
        if time_ylim is not None:
            axs_time[i].set_ylim(time_ylim)

        fft_spectrum = np.abs(np.fft.rfft(data[i]))
        axs_fft[i].plot(freqs, fft_spectrum)
        axs_fft[i].set_title(f"FFT Canal {i} {title_suffix}")
        axs_fft[i].set_xlabel("Fréquence (Hz)")
        axs_fft[i].set_ylabel("Amplitude")
        axs_fft[i].grid(True)
        axs_fft[i].set_xlim(freq_xlim)
        if fft_ylim is not None:
            axs_fft[i].set_ylim(fft_ylim)

    plt.tight_layout()
    return fig_time, fig_fft


############################################################################
# 2) Prétraitement EDF : sélection de canaux, filtrage, segmentation en epochs
############################################################################

def get_run_category(file_name):
    """
    Extrait le numéro de run depuis le nom du fichier et retourne la catégorie :
      - None pour baseline (run <=2)
      - "mains" pour runs dans [3,4,7,8,11,12]
      - "deux_effecteurs" pour runs dans [5,6,9,10,13,14]
    """
    m_obj = re.search(r'R(\d+)', file_name)
    if m_obj:
        run = int(m_obj.group(1))
        if run <= 2:
            return None
        elif run in [3, 4, 7, 8, 11, 12]:
            return "mains"
        elif run in [5, 6, 9, 10, 13, 14]:
            return "deux_effecteurs"
    return None


def process_edf(file_path, channels_to_keep, l_freq, h_freq, tmin=0.0, tmax=4.0):
    """
    Charge un EDF, sélectionne les canaux, applique le filtre et segmente en epochs.
    On conserve uniquement les événements T1 et T2.
    Pour garantir une durée constante, on élimine les epochs dont la taille ne correspond pas.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Limited 1 annotation")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick(channels_to_keep)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    selected_event_id = {key: val for key, val in event_id.items() if key in ['T1', 'T2']}
    if len(selected_event_id) == 0:
        print(f"[WARNING] Aucun événement T1/T2 dans {file_path}")
        return None

    epochs = mne.Epochs(raw, events, event_id=selected_event_id,
                        tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, verbose=False, on_missing='ignore')

    # Calcul du nombre de points attendus
    expected_n_times = int(round((tmax - tmin) * raw.info['sfreq'])) + 1
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    good_idx = [i for i in range(data.shape[0]) if data[i].shape[-1] == expected_n_times]
    if len(good_idx) < data.shape[0]:
        print(f"[INFO] Dropping {data.shape[0] - len(good_idx)} incomplete epochs from {file_path}")
        if len(good_idx) == 0:
            return None
        epochs = epochs[good_idx]
    return epochs


def process_subject(subject_dir, channels_to_keep, l_freq, h_freq, tmin, tmax):
    """
    Parcourt tous les EDF d'un sujet et répartit les epochs par catégorie.
    Retourne un dictionnaire : { 'mains': (X, y), 'deux_effecteurs': (X, y) }.
    Seules les catégories avec au moins 1 epoch sont retournées.
    """
    cat_epochs = {"mains": [], "deux_effecteurs": []}
    for file in os.listdir(subject_dir):
        if file.endswith(".edf"):
            cat = get_run_category(file)
            if cat is None:
                continue  # Ignorer les baselines
            file_path = os.path.join(subject_dir, file)
            epochs = process_edf(file_path, channels_to_keep, l_freq, h_freq, tmin, tmax)
            if epochs is not None and len(epochs) > 0:
                cat_epochs[cat].append(epochs)
                print(f"[INFO] {file}: {len(epochs)} epochs ajoutées à la catégorie {cat}")
            else:
                print(f"[INFO] {file}: 0 epoch (incomplet ou manquant) – ignoré.")

    out = {}
    for cat in cat_epochs:
        if len(cat_epochs[cat]) > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning,
                                        message=".*Concatenation of Annotations.*")
                try:
                    epochs_all = mne.concatenate_epochs(cat_epochs[cat])
                except ValueError:
                    print(f"[INFO] Aucune epoch valide pour la catégorie {cat} dans {os.path.basename(subject_dir)}")
                    continue
            X = epochs_all.get_data()  # (n_epochs, n_channels, n_times)
            y = epochs_all.events[:, 2]
            if X.size > 0:
                out[cat] = (X, y)
                print(f"[INFO] Sujet {os.path.basename(subject_dir)} - Catégorie {cat} : {X.shape[0]} epochs")
    if len(out) == 0:
        return None
    return out


############################################################################
# 3) Pipeline de modélisation : CSP + LDA
############################################################################

def build_pipeline(n_components=4):
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline([('CSP', csp), ('LDA', lda)])
    return pipeline


############################################################################
# 4) Découpage par sujet et agrégation des données par catégorie
############################################################################

def list_subject_dirs(eeg_dir):
    dirs = [os.path.join(eeg_dir, d) for d in os.listdir(eeg_dir)
            if os.path.isdir(os.path.join(eeg_dir, d)) and d.startswith("S") and len(d) == 4]
    dirs.sort()
    return dirs


def aggregate_subjects(subject_dirs, channels, l_freq, h_freq, tmin, tmax):
    """
    Agrège pour chaque catégorie les données de tous les sujets.
    Retourne un dict : { 'mains': (X, y), 'deux_effecteurs': (X, y) }.
    Seuls les sujets avec des epochs non vides sont pris en compte.

    Pour gérer les durées variables dans la dernière dimension (temps),
    on tronque chaque tableau à la longueur minimale trouvée parmi tous les sujets.
    """
    agg = {"mains": [], "deux_effecteurs": []}
    for subj_dir in subject_dirs:
        subj_data = process_subject(subj_dir, channels, l_freq, h_freq, tmin, tmax)
        if subj_data is not None:
            for cat in subj_data:
                if subj_data[cat][0].size > 0:
                    agg[cat].append(subj_data[cat])
                    print(f"[INFO] Sujet {os.path.basename(subj_dir)} ajouté à la catégorie {cat}")
                else:
                    print(f"[INFO] Sujet {os.path.basename(subj_dir)} a 0 epochs pour la catégorie {cat} – ignoré.")
    out = {}
    for cat in agg:
        if len(agg[cat]) > 0:
            X_list = [item[0] for item in agg[cat] if item[0].size > 0]
            y_list = [item[1] for item in agg[cat] if item[1].size > 0]
            if len(X_list) > 0:
                # Tronquer chaque tableau à la longueur minimale le long de l'axe temporel
                min_length = min(X.shape[-1] for X in X_list)
                X_list_fixed = [X[..., :min_length] for X in X_list]
                X_all = np.concatenate(X_list_fixed, axis=0)
                y_all = np.concatenate(y_list, axis=0)
                out[cat] = (X_all, y_all)
                print(f"[INFO] Agrégation finale pour {cat} : {X_all.shape[0]} epochs")
    return out


############################################################################
# 5) Script principal
############################################################################

# Configuration du backend matplotlib
matplotlib.use("TkAgg")

# Récupération du répertoire contenant les données EDF
eeg_dir = os.getenv("EEG_DIR", "")
print(f"[INFO] EEG DIR = {eeg_dir}")

# --- Affichage comparatif pour S001 (exemple) ---
subject_example = os.path.join(eeg_dir, "S001")
file_example = os.path.join(subject_example, "S001R01.edf")
raw_example = mne.io.read_raw_edf(file_example, preload=True)
print("[INFO] Exemple (brut) :", raw_example.info)

# Affichage du signal brut complet (64 canaux)
data_raw_example, times_raw_example = raw_example[:]
fig_time_raw, fig_fft_raw = display_eeg_signals_and_spectra(
    data=data_raw_example,
    times=times_raw_example,
    sfreq=raw_example.info['sfreq'],
    title_suffix="- Brut",
    time_ylim=None,
    freq_xlim=(0, 80),
    fft_ylim=(0, 0.03),
    grid_shape=(8, 8)  # 64 canaux
)

# Sélection des canaux sensorimoteurs
all_c_channels = [ch for ch in raw_example.info["ch_names"] if ch.startswith("C") and ch.endswith("..")]
if len(all_c_channels) < 3:
    raise ValueError("Moins de 3 canaux sensorimoteurs trouvés !")
channels_to_keep = all_c_channels[:3]
print(f"[INFO] Canaux sélectionnés : {channels_to_keep}")

# Prétraitement de l'exemple filtré (3 canaux) : filtre 8-32 Hz
raw_example_filtered = raw_example.copy()
raw_example_filtered.pick(channels_to_keep)
raw_example_filtered.filter(l_freq=8.0, h_freq=32.0, fir_design='firwin')
data_filt_example, times_filt_example = raw_example_filtered[:]

custom_margins = dict(left=0.038, bottom=0.05, right=0.983, top=0.97, wspace=0.2, hspace=0.256)
fig_time_filt, fig_fft_filt = display_eeg_signals_and_spectra(
    data=data_filt_example,
    times=times_filt_example,
    sfreq=raw_example.info['sfreq'],
    title_suffix="- Filtré",
    time_ylim=None,
    freq_xlim=(0, 80),
    fft_ylim=(0, 0.03),
    grid_shape=(1, 3),
    custom_margins=custom_margins
)

plt.show()

# --- Traitement global sur plusieurs sujets ---
subject_dirs = list_subject_dirs(eeg_dir)
print(f"[INFO] Nombre de sujets trouvés : {len(subject_dirs)}")

# Découpage global par sujet : 60% Training, 20% Test, 20% Holdout
train_dirs, temp_dirs = train_test_split(subject_dirs, test_size=0.4, random_state=42)
test_dirs, holdout_dirs = train_test_split(temp_dirs, test_size=0.5, random_state=42)
print(f"[INFO] Training subjects : {train_dirs}")
print(f"[INFO] Test subjects : {test_dirs}")
print(f"[INFO] Holdout subjects : {holdout_dirs}")

# Agrégation des données par catégorie (epochs de 4 s)
agg_train = aggregate_subjects(train_dirs, channels_to_keep, 8.0, 32.0, 0.0, 4.0)
agg_test = aggregate_subjects(test_dirs, channels_to_keep, 8.0, 32.0, 0.0, 4.0)
agg_holdout = aggregate_subjects(holdout_dirs, channels_to_keep, 8.0, 32.0, 0.0, 4.0)

for cat in agg_train:
    print(f"[INFO] Catégorie {cat} - X_train shape: {agg_train[cat][0].shape}")

############################################################################
# Entraînement de modèles spécialisés par catégorie et sauvegarde
############################################################################

models = {}
for cat in agg_train:
    print(f"[INFO] Traitement de la catégorie : {cat}")
    X_train, y_train = agg_train[cat]
    X_test, y_test = agg_test[cat]

    pipeline = build_pipeline(n_components=4)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"[INFO] Scores CV pour {cat} : {cv_scores}")
    print(f"[INFO] Accuracy moyenne CV pour {cat} : {np.mean(cv_scores):.2f}")

    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"[INFO] Accuracy sur Test set pour {cat} : {test_score:.2f}")

    models[cat] = pipeline
    model_filename = f"model_{cat}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[INFO] Modèle {cat} sauvegardé dans {model_filename}")

############################################################################
# Simulation de prédiction en streaming sur le jeu Holdout pour chaque catégorie
############################################################################

for cat in agg_holdout:
    X_holdout, y_holdout = agg_holdout[cat]
    print(f"[INFO] Prédictions sur le jeu Holdout pour la catégorie {cat} :")
    for i in range(min(10, X_holdout.shape[0])):
        # Utilisation de np.expand_dims pour s'assurer que l'input est un ndarray
        X_sample = np.expand_dims(X_holdout[i], axis=0)
        pred = models[cat].predict(X_sample)
        print(f"  Epoch {i} - Prédiction: {pred[0]} - Vérité: {y_holdout[i]}")
