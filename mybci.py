#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import warnings
import pickle
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from custom_mycsp import MyCSP

############################################################################
# Réglages de parallélisation BLAS / OpenMP
############################################################################
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"

############################################################################
# Constante minimale pour la longueur du signal pour le filtrage.
############################################################################
MIN_SIGNAL_LENGTH = 265

############################################################################
# 1) Classe FilterBankCSP
############################################################################
class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Implémentation du Filter Bank CSP.
    Pour chaque sous-bande (définie dans filter_bands), on entraîne un CSP séparé.
    Lors du transform, on filtre, on applique le CSP et on concatène les features.
    X doit avoir la forme (n_trials, n_channels, n_times).
    """
    def __init__(self,
                 filter_bands=[(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)],
                 n_components=4,
                 csp_reg='ledoit_wolf',
                 csp_log=True,
                 sfreq=160.0,
                 n_jobs=16):
        self.filter_bands = filter_bands
        self.n_components = n_components
        self.csp_reg = csp_reg
        self.csp_log = csp_log
        self.sfreq = sfreq
        self.n_jobs = n_jobs
        self._csps = []  # liste des tuples (l_freq, h_freq, CSP)

    def _pad_if_needed(self, X):
        n_times = X.shape[-1]
        if n_times < MIN_SIGNAL_LENGTH:
            pad_width = MIN_SIGNAL_LENGTH - n_times
            X_padded = np.pad(X, pad_width=((0, 0), (0, 0), (0, pad_width)), mode='constant')
            return X_padded, n_times
        else:
            return X, None

    def fit(self, X, y):
        import mne.filter
        self._csps = []
        X_use, orig_length = self._pad_if_needed(X)
        for (l_freq, h_freq) in self.filter_bands:
            X_filt_full = mne.filter.filter_data(data=X_use, sfreq=self.sfreq,
                                                 l_freq=l_freq, h_freq=h_freq,
                                                 verbose=False, n_jobs=self.n_jobs)
            X_filt = X_filt_full[..., :orig_length] if orig_length is not None else X_filt_full
            csp = MyCSP(n_components=self.n_components, reg=self.csp_reg,
                        norm_trace=False, log=self.csp_log)
            csp.fit(X_filt, y)
            self._csps.append((l_freq, h_freq, csp))
        return self

    def transform(self, X):
        import mne.filter
        X_use, orig_length = self._pad_if_needed(X)
        X_features_list = []
        for (l_freq, h_freq, csp) in self._csps:
            X_filt_full = mne.filter.filter_data(data=X_use, sfreq=self.sfreq,
                                                 l_freq=l_freq, h_freq=h_freq,
                                                 verbose=False, n_jobs=self.n_jobs)
            X_filt = X_filt_full[..., :orig_length] if orig_length is not None else X_filt_full
            X_csp = csp.transform(X_filt)
            X_features_list.append(X_csp)
        return np.concatenate(X_features_list, axis=1)

############################################################################
# Nouvelle fonction pour l'affichage du PSD
############################################################################
from mne.time_frequency import psd_array_welch

def plot_psd_all_channels_0_80(raw, n_fft=2048, title="PSD all channels (0-80 Hz)"):
    """
    Affiche le spectre de densité de puissance (PSD) de 0 à 80 Hz
    pour chaque canal EEG.
    """
    data = raw.get_data()  # (n_channels, n_times)
    sfreq = raw.info['sfreq']

    # PSD canal par canal
    psds, freqs = psd_array_welch(
        data,  # data shape (n_channels, n_times)
        sfreq=sfreq,
        fmin=0,
        fmax=80,
        window='hann',
        n_fft=2048,  # ou n_fft=1024
        n_per_seg=512,  # réduit la résolution
        n_overlap=256,  # pour un overlap de 50%
        average='mean',
        verbose=False
    )
    psds_db = 10 * np.log10(psds)  # passage en dB

    plt.figure(figsize=(10, 6))
    for ch_idx in range(psds_db.shape[0]):
        plt.plot(freqs, psds_db[ch_idx], linewidth=0.5, alpha=0.7)

    plt.title(title)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel(r"PSD (dB)")
    plt.xlim([0, 80])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


from scipy.ndimage import gaussian_filter1d

def plot_psd_all_channels_0_80(raw, n_fft=2048, title="PSD (0–80 Hz)"):
    """
    Calcule et affiche la PSD de 0 à 80 Hz pour chaque canal EEG.
    """
    data = raw.get_data()            # shape (n_channels, n_times)
    sfreq = raw.info['sfreq']

    # Calcul de la PSD entre 0 et 80 Hz
    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0, fmax=80,
                                  n_fft=n_fft, verbose=False)
    psds_db = 10 * np.log10(psds)    # passage en dB

    plt.figure(figsize=(10, 6))
    for ch_idx in range(psds_db.shape[0]):
        plt.plot(freqs, psds_db[ch_idx], linewidth=0.5, alpha=0.7)

    plt.title(title)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("PSD (dB)")
    plt.xlim([0, 80])               # Echelle 0–80 Hz
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_mean_psd_with_std_0_80(raw, n_fft=2048, smooth_sigma=1, title="PSD moyen (0–80 Hz)"):
    """
    Calcule la PSD (0–80 Hz) pour chaque canal, puis affiche la courbe moyenne
    (lissée) et la zone ±1 écart-type.
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0, fmax=80,
                                  n_fft=n_fft, verbose=False)
    psds_db = 10 * np.log10(psds)  # (n_channels, n_freqs)

    # Moyenne et std sur l’axe 0 (canaux)
    mean_psd = np.mean(psds_db, axis=0)
    std_psd  = np.std(psds_db, axis=0)

    # Lissage (optionnel) des courbes
    if smooth_sigma is not None and smooth_sigma > 0:
        mean_psd = gaussian_filter1d(mean_psd, sigma=smooth_sigma)
        std_psd  = gaussian_filter1d(std_psd,  sigma=smooth_sigma)

    plt.figure(figsize=(8, 5))
    # Courbe moyenne
    plt.plot(freqs, mean_psd, color='black', label='Moyenne')

    # Zone ±1 écart-type
    plt.fill_between(freqs,
                     mean_psd - std_psd,
                     mean_psd + std_psd,
                     color='gray', alpha=0.4, label='±1 écart-type')

    plt.title(title)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("PSD (dB)")
    plt.xlim([0, 80])  # Même échelle de 0 à 80 Hz
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()



def plot_psd_all_channels(raw, fmin=0, fmax=80, n_fft=2048, title="EEG (64 canaux)"):
    """
    Affiche le spectre de densité de puissance (PSD) pour chaque canal EEG,
    en utilisant la méthode 'welch' via raw.compute_psd().
    """
    # On calcule le PSD par canal
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, method='welch', n_fft=n_fft)

    # psd.get_data() renvoie un array de shape (n_channels, n_freqs)
    psds = psd.get_data()
    freqs = psd.freqs

    # Passage en dB
    psds_db = 10 * np.log10(psds)

    plt.figure(figsize=(10, 8))
    for channel_idx in range(psds_db.shape[0]):
        plt.plot(freqs, psds_db[channel_idx], linewidth=0.5, alpha=0.7)

    plt.xlabel('Fréquence (Hz)')
    plt.ylabel(r'$\mu V^2 / Hz$ (dB)')
    plt.title(title)
    plt.xlim([fmin, fmax])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_combined_psd(raw, fmin=0, fmax=80, n_fft=2048, title="EEG (64 canaux)"):
    """
    Calcule et affiche la densité spectrale de puissance (PSD) en dB,
    moyennée sur tous les canaux EEG.
    Utilise psd_array_welch pour compatibilité avec d'anciennes versions de MNE.
    """
    from mne.time_frequency import psd_array_welch
    data = raw.get_data()  # shape: (n_channels, n_times)
    sfreq = raw.info['sfreq']

    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                  n_fft=n_fft, verbose=False)
    # Moyenne sur les canaux, passage en dB
    psds_mean = 10 * np.log10(psds.mean(axis=0))

    plt.figure()
    plt.plot(freqs, psds_mean, color='black')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('PSD (dB)')
    plt.title(title)
    plt.xlim([fmin, fmax])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

############################################################################
# 3) Gestion des runs et catégorisation en 6 expériences
############################################################################
def get_run_category(file_name):
    """
    Retourne une étiquette d'expérience selon le numéro de run :
    - R1,R2: Baseline (ignoré)
    - R3,R4: "exp0" - Tâches unilatérales (mouvement de main ou imagerie)
    - R5,R6: "exp1" - Tâches bilatérales (mouvement de mains ou pieds, réel/imagé)
    - R7,R8: "exp2" - Répétition des tâches unilatérales
    - R9,R10: "exp3" - Répétition des tâches bilatérales
    - R11,R12: "exp4" - Nouvelle répétition des tâches unilatérales
    - R13,R14: "exp5" - Nouvelle répétition des tâches bilatérales
    """
    m_obj = re.search(r'R(\d+)', file_name)
    if m_obj:
        run = int(m_obj.group(1))
        if run <= 2:
            return None  # on ignore les baselines
        if run in [3, 4]:
            return "exp0"
        elif run in [5, 6]:
            return "exp1"
        elif run in [7, 8]:
            return "exp2"
        elif run in [9, 10]:
            return "exp3"
        elif run in [11, 12]:
            return "exp4"
        elif run in [13, 14]:
            return "exp5"
    return None

############################################################################
# 4) Chargement EDF, filtrage et découpage en epochs
############################################################################
def process_edf(file_path, channels_to_keep, l_freq, h_freq, tmin=0.5, tmax=2.5, do_reref=True):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Limited 1 annotation")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick(channels_to_keep)
    if do_reref:
        raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    selected_event_id = {k: v for k, v in event_id.items() if k in ['T0', 'T1', 'T2']}
    if len(selected_event_id) == 0:
        print(f"[WARNING] Aucun événement T0/T1/T2 dans {file_path}")
        return None
    epochs = mne.Epochs(raw, events, event_id=selected_event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False, on_missing='ignore')
    sfreq = raw.info['sfreq']
    expected_n_times = int(round((tmax - tmin) * sfreq)) + 1
    data = epochs.get_data()
    good_idx = [i for i in range(data.shape[0]) if data[i].shape[-1] == expected_n_times]
    if len(good_idx) < data.shape[0]:
        if len(good_idx) == 0:
            print(f"[INFO] All epochs incomplete => discard {file_path}")
            return None
        epochs = epochs[good_idx]
    return epochs

############################################################################
# 5) Traitement d'un sujet complet
############################################################################
def process_subject(subject_dir, channels_to_keep, l_freq, h_freq, tmin, tmax):
    """
    Parcourt les fichiers .edf du dossier subject_dir et regroupe les epochs par
    expérience (exp0 à exp5). Les runs baseline (R1 et R2) sont ignorés.
    Pour chaque run non-baseline, on garde uniquement les epochs dont l'événement est T1 ou T2.
    """
    cat_epochs = {"exp0": [], "exp1": [], "exp2": [], "exp3": [], "exp4": [], "exp5": []}
    for f in os.listdir(subject_dir):
        if f.endswith(".edf"):
            exp = get_run_category(f)
            if exp is None:
                continue
            file_path = os.path.join(subject_dir, f)
            epochs = process_edf(file_path, channels_to_keep, l_freq, h_freq, tmin, tmax, do_reref=True)
            if epochs is not None and len(epochs) > 0:
                data = epochs.get_data()
                events = epochs.events[:, 2]
                idx = (events == 1) | (events == 2)
                if not np.any(idx):
                    continue
                data = data[idx]
                labels = events[idx]
                cat_epochs[exp].append((data, labels))
                print(f"[INFO] {f}: {len(data)} epochs -> {exp}")
            else:
                print(f"[INFO] {f}: 0 epoch -> ignoré.")
    out = {}
    for cat in cat_epochs:
        if len(cat_epochs[cat]) == 0:
            continue
        min_length = min(x[0].shape[-1] for x in cat_epochs[cat])
        X_list, y_list = [], []
        for (X_c, y_c) in cat_epochs[cat]:
            X_list.append(X_c[..., :min_length])
            y_list.append(y_c)
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        out[cat] = (X_all, y_all)
        print(f"[INFO] Subject {os.path.basename(subject_dir)} - {cat}: {X_all.shape[0]} epochs")
    return out if len(out) > 0 else None

############################################################################
# 6) Agrégation multi-sujets
############################################################################
def list_subject_dirs(eeg_dir):
    dirs = []
    for d in os.listdir(eeg_dir):
        path = os.path.join(eeg_dir, d)
        if os.path.isdir(path) and d.startswith("S") and len(d) == 4:
            dirs.append(d)
    dirs.sort()
    return [os.path.join(eeg_dir, d) for d in dirs]

def aggregate_subjects(subject_dirs, channels, l_freq, h_freq, tmin, tmax):
    agg = {"exp0": [], "exp1": [], "exp2": [], "exp3": [], "exp4": [], "exp5": []}
    for sd in subject_dirs:
        data_sub = process_subject(sd, channels, l_freq, h_freq, tmin, tmax)
        if data_sub is not None:
            for cat in data_sub:
                Xc, yc = data_sub[cat]
                if Xc.size > 0:
                    agg[cat].append((Xc, yc))
    out = {}
    for cat in agg:
        if len(agg[cat]) == 0:
            continue
        min_length = min(x[0].shape[-1] for x in agg[cat])
        X_list, y_list = [], []
        for (X_c, y_c) in agg[cat]:
            X_list.append(X_c[..., :min_length])
            y_list.append(y_c)
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        out[cat] = (X_all, y_all)
        print(f"[INFO] Cat={cat}: total epochs={X_all.shape[0]}, time={min_length}")
    return out

############################################################################
# 7) Construction du pipeline FBCSP + LDA
############################################################################
def build_pipeline_fbcsp():
    filter_bands = [(8, 11), (11, 13), (13, 20), (20, 26), (26, 32)]
    fbcsp = FilterBankCSP(filter_bands=filter_bands,
                          n_components=11,
                          csp_reg='ledoit_wolf',
                          csp_log=True,
                          sfreq=160.0,
                          n_jobs=16)
    clf = LDA(solver='lsqr', shrinkage='auto')
    return Pipeline([('FBCSP', fbcsp), ('LDA', clf)])

############################################################################
# 8) Fonction de segmentation par glissement
############################################################################
def segment_epochs_sliding(X, window_fraction=0.5, step_fraction=0.25):
    """
    Découpe chaque epoch en segments via un sliding window.
    - X : array de forme (n_epochs, n_channels, n_times)
    - window_fraction : fraction de la durée de l'epoch pour la fenêtre (ex: 0.5)
    - step_fraction : fraction du nombre d'échantillons pour le pas (ex: 0.25)
    Retourne un array de segments de forme (n_new_epochs, n_channels, window_length).
    """
    n_epochs, n_channels, n_times = X.shape
    window_length = int(n_times * window_fraction)
    step = int(n_times * step_fraction)
    segments = []
    for i in range(n_epochs):
        for start in range(0, n_times - window_length + 1, step):
            segments.append(X[i, :, start:start+window_length])
    return np.array(segments)

############################################################################
# 9) Main
############################################################################
if __name__ == "__main__":
    eeg_dir = os.getenv("EEG_DIR", "")
    print(f"[INFO] EEG_DIR={eeg_dir}")

    # Variables communes
    chosen_channels = ['C3..', 'Cz..', 'C4..', 'C1..', 'C2..', 'Fcz.', 'Fc3.', 'Fc4.',
                       'Cpz.', 'Cp3.', 'Cp4.', 'Cp1.', 'Cp2.', 'Pz..', 'Fz..']
    l_freq = 1.0
    h_freq = 40.0
    tmin, tmax = 0.7, 3.5

    # Harmonisation des arguments :
    # - Si 1 argument : mode multi-sujets (ex: "train" ou "predict")
    # - Si 3 arguments : mode individuel avec <subject_id> <run_id> <mode>
    if len(sys.argv) == 1:
        sys.exit(0)

    elif len(sys.argv) == 2:
        mode_main = sys.argv[1].lower()
        if mode_main == "train":
            # Mode multi-sujets : entraînement des 6 modèles
            start_time = time.time()
            subject_dirs = list_subject_dirs(eeg_dir)
            print(f"[INFO] {len(subject_dirs)} sujets trouvés.")

            # Exemple de split train/test/holdout
            train_dirs, tmp_dirs = train_test_split(subject_dirs, test_size=0.4, random_state=42)
            test_dirs, holdout_dirs = train_test_split(tmp_dirs, test_size=0.5, random_state=42)
            print(f"[INFO] Train: {train_dirs}")
            print(f"[INFO] Test: {test_dirs}")
            print(f"[INFO] Holdout: {holdout_dirs}")

            agg_train = aggregate_subjects(train_dirs, chosen_channels, l_freq, h_freq, tmin, tmax)
            agg_test = aggregate_subjects(test_dirs, chosen_channels, l_freq, h_freq, tmin, tmax)
            agg_hold = aggregate_subjects(holdout_dirs, chosen_channels, l_freq, h_freq, tmin, tmax)

            categories = ["exp0", "exp1", "exp2", "exp3", "exp4", "exp5"]
            models = {}
            for cat in categories:
                if cat not in agg_train:
                    print(f"[WARN] Pas de data train pour cat={cat}, skip.")
                    continue
                if cat not in agg_test:
                    print(f"[WARN] Pas de data test pour cat={cat}, skip.")
                    continue
                X_train, y_train = agg_train[cat]
                X_test, y_test = agg_test[cat]
                print(f"\n[INFO] ============ Catégorie {cat} ============")
                print(f"[INFO] X_train={X_train.shape}, y_train={y_train.shape}")
                if len(np.unique(y_train)) < 2:
                    print(f"[WARN] Catégorie {cat} n'a qu'une seule classe en train, skip.")
                    continue
                pipeline = build_pipeline_fbcsp()
                try:
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
                    print(f"[INFO] cross-value-scores={cv_scores}, mean={cv_scores.mean():.2f}")
                except Exception as e:
                    print(f"[ERROR] Erreur cross_val pour {cat}: {e}")
                    continue
                pipeline.fit(X_train, y_train)
                test_acc = pipeline.score(X_test, y_test)
                print(f"[INFO] Test accuracy={test_acc:.2f}")
                models[cat] = pipeline
                model_filename = f"model_{cat}.pkl"
                with open(model_filename, "wb") as ff:
                    pickle.dump(pipeline, ff)
                print(f"[INFO] Modèle sauvegardé => {model_filename}")
            holdout_accuracies = []
            for cat in models:
                if cat not in agg_hold:
                    print(f"[INFO] cat={cat}, pas de data holdout => skip.")
                    continue
                X_hold, y_hold = agg_hold[cat]
                print(f"\n[INFO] HOLDOUT prédictions cat={cat}, shape={X_hold.shape}")
                pipeline = models[cat]
                preds = pipeline.predict(X_hold)
                hold_acc = np.mean(preds == y_hold)
                print(f"[INFO] Holdout accuracy={hold_acc:.2f}")
                holdout_accuracies.append(hold_acc)
                for i in range(min(10, len(preds))):
                    print(f"Epoch {i} => pred={preds[i]}, true={y_hold[i]}")
            if holdout_accuracies:
                moyenne = np.mean(holdout_accuracies)
                print(f"\n[INFO] Moyenne holdout accuracy des modèles : {moyenne:.3f}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print(f"\n[INFO] Temps total d'exécution : {minutes}:{seconds:02d} (min:ss)")

        elif mode_main == "predict":
            sys.exit("[INFO] Mode multi-sujets predict non implémenté dans cette version.")

        elif mode_main == "analyse":
            # Mode analyse par défaut individuel : utilisation de S001 et S001R03
            test_subj = "S001"
            test_run = "S001R03.edf"
            path_example = os.path.join(eeg_dir, test_subj, test_run)
            try:
                raw_example = mne.io.read_raw_edf(path_example, preload=True)
            except Exception as e:
                sys.exit(f"[ERROR] Impossible de charger {path_example}: {e}")

            print("[INFO] Affichage des spectres PSD individuels pour les 64 canaux EEG...")

            raw_example.plot(
                n_channels=64,
                duration=20,
                scalings='auto',
                title='Données brutes (64 canaux)',
                show=True,
                block=True
            )
            # raw_example.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)

            plot_psd_all_channels_0_80(raw_example, n_fft=2048, title="PSD (0-80 Hz) - tous canaux")

            raw_8_32 = raw_example.copy().filter(l_freq=8, h_freq=32, fir_design='firwin', verbose=False)

            plot_psd_all_channels_0_80(raw_8_32, n_fft=2048,
                                       title="(3) Filtré 8–32 Hz, PSD 0–80 Hz")


            plot_mean_psd_with_std_0_80(raw_8_32, n_fft=2048, smooth_sigma=8,
                                        title="(2) Non filtré, PSD 0–80 Hz (moy + std)")



        else:
            sys.exit("[ERROR] Mode inconnu pour le mode multi-sujets.")
        sys.exit(0)

    elif len(sys.argv) == 4:
        # Mode individuel : les arguments sont <subject_id> <run_id> <mode>
        subject_id = sys.argv[1]
        run_id = sys.argv[2]
        mode_indiv = sys.argv[3].lower()  # "analyse", "train" ou "predict"
        subject_folder = "S" + subject_id.zfill(3)
        file_name = f"{subject_folder}R{run_id.zfill(2)}.edf"
        full_path = os.path.join(eeg_dir, subject_folder, file_name)
        if mode_indiv == "a":
            mode_indiv = "analyse"
        if mode_indiv == "p":
            mode_indiv = "predict"
        if mode_indiv == "t":
            mode_indiv = "train"
        print(f"[INFO] Mode {mode_indiv} pour le sujet {subject_folder} et le run {run_id}")

        if int(run_id) <= 2:
            sys.exit("[ERROR] Les runs baseline (1 et 2) sont ignorés dans ce mode.")

        if mode_indiv == "analyse":
            # Analyse simple du PSD sur le run donné
            try:
                raw_example = mne.io.read_raw_edf(full_path, preload=True)
            except Exception as e:
                sys.exit(f"[ERROR] Impossible de charger {full_path}: {e}")

            raw_example.plot(
                n_channels=64,
                duration=20,
                scalings='auto',
                title='Données brutes (64 canaux)',
                show=True,
                block=True
            )

            plot_psd_all_channels_0_80(raw_example, n_fft=2048, title="PSD (0-80 Hz) - tous canaux")

            raw_8_32 = raw_example.copy().filter(l_freq=8, h_freq=32, fir_design='firwin', verbose=False)

            plot_psd_all_channels_0_80(raw_8_32, n_fft=2048,
                                       title="(3) Filtré 8–32 Hz, PSD 0–80 Hz")


            plot_mean_psd_with_std_0_80(raw_8_32, n_fft=2048, smooth_sigma=8,
                                        title="(2) Non filtré, PSD 0–80 Hz (moy + std)")

        elif mode_indiv == "train":
            # Entraînement individuel
            epochs = process_edf(full_path, chosen_channels, l_freq, h_freq, tmin, tmax)
            if epochs is None:
                sys.exit("[ERROR] Aucun epoch trouvé dans le fichier.")
            events_all = epochs.events[:, 2]
            idx = (events_all == 1) | (events_all == 2)
            if not np.any(idx):
                sys.exit("[ERROR] Aucune epoch avec T1/T2 dans le fichier.")
            X_orig = epochs.get_data()[idx]
            y_orig = events_all[idx]
            # Découpage en segments via sliding window (plus de marge)
            X_new = segment_epochs_sliding(X_orig, window_fraction=0.5, step_fraction=0.25)
            # Chaque segment hérite du label de l'epoch d'origine
            y_new = []
            n_epochs, _, _ = X_orig.shape
            window_length = int(X_orig.shape[2] * 0.5)
            step = int(X_orig.shape[2] * 0.25)
            n_segments_per_epoch = ((X_orig.shape[2] - window_length) // step) + 1
            for i in range(len(y_orig)):
                y_new.extend([y_orig[i]] * n_segments_per_epoch)
            X_new = np.array(X_new)
            y_new = np.array(y_new)
            from sklearn.model_selection import train_test_split
            X_train, X_holdout, y_train, y_holdout = train_test_split(X_new, y_new, test_size=0.40, random_state=43)
            try:
                cv_scores = cross_val_score(build_pipeline_fbcsp(), X_train, y_train, cv=10, scoring='accuracy')
                print(f"cross-value-scores: {cv_scores}, mean: {cv_scores.mean():.4f}")
            except Exception as e:
                print(f"[ERROR] Erreur lors du cross_val_score: {e}")

            pipeline = build_pipeline_fbcsp()
            pipeline.fit(X_train, y_train)
            model_filename = f"model_{subject_folder}R{run_id}.pkl"
            holdout_filename = f"model_{subject_folder}R{run_id}_holdout.pkl"
            with open(model_filename, "wb") as ff:
                pickle.dump(pipeline, ff)
            with open(holdout_filename, "wb") as ff:
                pickle.dump({"X_holdout": X_holdout, "y_holdout": y_holdout}, ff)
            print(f"[INFO] Modèle sauvegardé => {model_filename}")
            print(f"[INFO] Données holdout sauvegardées => {holdout_filename}")

        elif mode_indiv == "predict":
            # Prédiction individuelle
            epochs = process_edf(full_path, chosen_channels, l_freq, h_freq, tmin, tmax)
            if epochs is None:
                sys.exit("[ERROR] Aucun epoch trouvé dans le fichier.")
            events_all = epochs.events[:, 2]
            idx = (events_all == 1) | (events_all == 2)
            if not np.any(idx):
                sys.exit("[ERROR] Aucune epoch avec T1/T2 dans le fichier.")
            X_orig = epochs.get_data()[idx]
            y_orig = events_all[idx]
            X_new = segment_epochs_sliding(X_orig, window_fraction=0.5, step_fraction=0.25)
            y_new = []
            n_epochs, _, _ = X_orig.shape
            window_length = int(X_orig.shape[2] * 0.5)
            step = int(X_orig.shape[2] * 0.25)
            n_segments_per_epoch = ((X_orig.shape[2] - window_length) // step) + 1
            for i in range(len(y_orig)):
                y_new.extend([y_orig[i]] * n_segments_per_epoch)
            X_new = np.array(X_new)
            y_new = np.array(y_new)
            # On ne retient que 8 segments max pour la "simulation temps réel"
            if X_new.shape[0] >= 8:
                X_holdout = X_new[:8]
                y_holdout = y_new[:8]
            else:
                X_holdout = X_new
                y_holdout = y_new
            model_filename = f"model_{subject_folder}R{run_id}.pkl"
            try:
                with open(model_filename, "rb") as ff:
                    pipeline = pickle.load(ff)
            except Exception as e:
                sys.exit(f"[ERROR] Impossible de charger le modèle: {e}")

            print("[INFO] Début de la prédiction en mode simulation temps réel sur les données holdout:")
            correct_count = 0
            for i, (epoch, true_label) in enumerate(zip(X_holdout, y_holdout)):
                pred = pipeline.predict(epoch[None, ...])[0]
                equal_str = "True" if pred == true_label else "False"
                print(f"Epoch {i:02d}: [{pred}] [{true_label}] {equal_str}")
                if pred == true_label:
                    correct_count += 1
                time.sleep(0.5)
            final_acc = correct_count / len(y_holdout) if len(y_holdout) > 0 else 0.0
            print(f"Accuracy: {final_acc:.4f}")

        else:
            sys.exit("[ERROR] Mode inconnu. Utilisez 'analyse', 'train' ou 'predict'.")

    else:
        sys.exit("[ERROR] Nombre d'arguments incorrect. Utilisez soit 1 argument pour multi-sujets, soit 3 pour individuel.")
