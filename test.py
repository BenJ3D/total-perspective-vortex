#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

############################################################################
# Réglages de parallélisation BLAS / OpenMP
# Sur un AMD 5900X (12 coeurs/24 threads), par exemple on peut mettre 16.
############################################################################
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

import re
import warnings
import pickle
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline


############################################################################
# Constante minimale pour la longueur du signal pour le filtrage.
# (En dessous de cette taille, on "pad" pour appliquer le filtrage FIR).
############################################################################
MIN_SIGNAL_LENGTH = 265


############################################################################
# 1) Classe FilterBankCSP : applique plusieurs bandes de fréquences, CSP, puis concatène
############################################################################

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Implémentation du Filter Bank CSP.
    Pour chaque sous-bande (définie dans filter_bands), on entraîne un CSP séparé.
    Lors du transform, on filtre, on applique le CSP et on concatène les features.
    X est supposé avoir la forme (n_trials, n_channels, n_times).

    Paramètres notables :
      - n_jobs : nombre de coeurs (threads) à utiliser pour mne.filter.filter_data.
    """

    def __init__(
        self,
        filter_bands=[(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)],
        n_components=4,
        csp_reg='ledoit_wolf',
        csp_log=True,
        sfreq=160.0,
        n_jobs=16   # << ajout d'un paramètre pour le filtrage multi-thread
    ):
        self.filter_bands = filter_bands
        self.n_components = n_components
        self.csp_reg = csp_reg
        self.csp_log = csp_log
        self.sfreq = sfreq
        self.n_jobs = n_jobs
        self._csps = []  # liste des (l_freq, h_freq, CSP)

    def _pad_if_needed(self, X):
        """
        Si la longueur temporelle de X est < MIN_SIGNAL_LENGTH,
        on applique un zero-padding à la fin pour pouvoir filtrer sans souci.
        """
        n_times = X.shape[-1]
        if n_times < MIN_SIGNAL_LENGTH:
            pad_width = MIN_SIGNAL_LENGTH - n_times
            X_padded = np.pad(
                X, pad_width=((0, 0), (0, 0), (0, pad_width)),
                mode='constant'
            )
            return X_padded, n_times  # on gardera trace de la longueur "originale"
        else:
            return X, None

    def fit(self, X, y):
        import mne.filter
        self._csps = []

        # Eventuel padding pour éviter erreur si n_times < 265
        X_use, orig_length = self._pad_if_needed(X)

        for (l_freq, h_freq) in self.filter_bands:
            # Filtrage dans la sous-bande
            X_filt_full = mne.filter.filter_data(
                data=X_use,
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                verbose=False,
                n_jobs=self.n_jobs
            )
            # Si on avait padé, on recadre à la longueur originale
            if orig_length is not None:
                X_filt = X_filt_full[..., :orig_length]
            else:
                X_filt = X_filt_full

            # Création et entraînement du CSP
            csp = CSP(
                n_components=self.n_components,
                reg=self.csp_reg,
                norm_trace=False,
                log=self.csp_log
            )
            csp.fit(X_filt, y)
            self._csps.append((l_freq, h_freq, csp))

        return self

    def transform(self, X):
        import mne.filter
        X_use, orig_length = self._pad_if_needed(X)

        X_features_list = []
        for (l_freq, h_freq, csp) in self._csps:
            # Filtre la même sous-bande
            X_filt_full = mne.filter.filter_data(
                data=X_use,
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                verbose=False,
                n_jobs=self.n_jobs
            )
            if orig_length is not None:
                X_filt = X_filt_full[..., :orig_length]
            else:
                X_filt = X_filt_full

            # Projection CSP -> features
            X_csp = csp.transform(X_filt)
            X_features_list.append(X_csp)

        # Concatène les features de toutes les sous-bandes
        return np.concatenate(X_features_list, axis=1)


############################################################################
# 2) Fonctions d'affichage (EEG temps + FFT)
############################################################################

def display_eeg_signals_and_spectra(
    data, times, sfreq, title_suffix="",
    time_ylim=None, freq_xlim=(0, 100),
    fft_ylim=None, grid_shape=None,
    custom_margins=None
):
    """
    Affiche pour chaque canal le signal EEG temporel et son spectre FFT.
    """
    import math

    n_channels = data.shape[0]
    if grid_shape is None:
        n_rows = int(math.ceil(math.sqrt(n_channels)))
        n_cols = int(math.ceil(n_channels / n_rows))
    else:
        n_rows, n_cols = grid_shape

    fig_time, axs_time = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs_time = np.array(axs_time).reshape(-1) if n_channels > 1 else [axs_time]

    fig_fft, axs_fft = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs_fft = np.array(axs_fft).reshape(-1) if n_channels > 1 else [axs_fft]

    if custom_margins is not None:
        fig_time.subplots_adjust(**custom_margins)
        fig_fft.subplots_adjust(**custom_margins)
    else:
        fig_time.subplots_adjust(left=0.07, right=0.986, top=0.967, bottom=0.06,
                                 wspace=0.2, hspace=0.2)
        fig_fft.subplots_adjust(left=0.07, right=0.986, top=0.967, bottom=0.06,
                                wspace=0.2, hspace=0.2)

    freqs = np.fft.rfftfreq(len(times), d=1 / sfreq)
    for i in range(n_channels):
        # Tracé temporel
        axs_time[i].plot(times, data[i])
        axs_time[i].set_title(f"Canal {i} {title_suffix}")
        axs_time[i].set_xlabel("Temps (s)")
        axs_time[i].set_ylabel("Amplitude")
        axs_time[i].grid(True)
        if time_ylim is not None:
            axs_time[i].set_ylim(time_ylim)

        # Tracé FFT
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
# 3) Gestion des runs et détermination de catégorie
############################################################################

def get_run_category(file_name):
    """
    Détermine la catégorie (mains_reel, mains_image, deux_reel, deux_image)
    en fonction du numéro de run dans le nom de fichier :
        R3, R7, R11 -> mains_reel
        R4, R8, R12 -> mains_image
        R5, R9, R13 -> deux_reel
        R6, R10, R14 -> deux_image
    Retourne None pour les runs <= 2 (baselines).
    """
    m_obj = re.search(r'R(\d+)', file_name)
    if m_obj:
        run = int(m_obj.group(1))
        if run <= 2:
            return None
        if run in [3, 7, 11]:
            return "mains_reel"
        elif run in [4, 8, 12]:
            return "mains_image"
        elif run in [5, 9, 13]:
            return "deux_reel"
        elif run in [6, 10, 14]:
            return "deux_image"
    return None


############################################################################
# 4) Chargement EDF, filtrage, découpage en epochs
############################################################################

def process_edf(file_path, channels_to_keep, l_freq, h_freq, tmin=0.5, tmax=2.5, do_reref=True):
    """
    Charge un fichier EDF, conserve channels_to_keep, re-référence moyenne,
    filtre [l_freq, h_freq], et découpe en epochs [tmin, tmax].
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Limited 1 annotation")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    raw.pick(channels_to_keep)
    if do_reref:
        raw.set_eeg_reference('average', projection=False, verbose=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    selected_event_id = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
    if len(selected_event_id) == 0:
        print(f"[WARNING] Aucun événement T1/T2 dans {file_path}")
        return None

    epochs = mne.Epochs(
        raw, events, event_id=selected_event_id,
        tmin=tmin, tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
        on_missing='ignore'
    )

    sfreq = raw.info['sfreq']
    expected_n_times = int(round((tmax - tmin) * sfreq)) + 1
    data = epochs.get_data()
    # On vérifie que chaque epoch a la taille temporelle attendue
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
    Parcourt les .edf du dossier subject_dir, détermine la catégorie,
    puis empile toutes les epochs de cette catégorie.
    Retourne un dict : {
        'mains_reel': (X, y),
        'mains_image': (X, y),
        'deux_reel': (X, y),
        'deux_image': (X, y)
    }
    """
    cat_epochs = {
        "mains_reel": [],
        "mains_image": [],
        "deux_reel": [],
        "deux_image": []
    }

    for f in os.listdir(subject_dir):
        if f.endswith(".edf"):
            cat = get_run_category(f)
            if cat is not None:  # On ne traite pas les baselines R1,R2
                file_path = os.path.join(subject_dir, f)
                epochs = process_edf(
                    file_path, channels_to_keep,
                    l_freq, h_freq, tmin, tmax,
                    do_reref=True
                )
                if epochs is not None and len(epochs) > 0:
                    cat_epochs[cat].append(epochs)
                    print(f"[INFO] {f}: {len(epochs)} epochs -> {cat}")
                else:
                    print(f"[INFO] {f}: 0 epoch -> ignoré.")

    out = {}
    for cat in cat_epochs:
        if len(cat_epochs[cat]) > 0:
            try:
                all_ep = mne.concatenate_epochs(cat_epochs[cat])
            except ValueError:
                print(f"[INFO] Aucune epoch valide pour cat={cat} dans {subject_dir}")
                continue
            X = all_ep.get_data()       # shape=(n_epochs, n_channels, n_times)
            y = all_ep.events[:, 2]    # T1=1, T2=2
            if X.size > 0:
                out[cat] = (X, y)
                print(f"[INFO] Subject {os.path.basename(subject_dir)} - {cat}: {X.shape[0]} epochs")

    return out if len(out) > 0 else None


############################################################################
# 6) Agrégation multi-sujets
############################################################################

def list_subject_dirs(eeg_dir):
    """
    Retourne la liste des dossiers SXXX dans eeg_dir
    """
    dirs = []
    for d in os.listdir(eeg_dir):
        path = os.path.join(eeg_dir, d)
        if os.path.isdir(path) and d.startswith("S") and len(d) == 4:
            dirs.append(path)
    dirs.sort()
    return dirs


def aggregate_subjects(subject_dirs, channels, l_freq, h_freq, tmin, tmax):
    """
    Agrège pour chaque catégorie : on concatène tous les X, y.
    On tronque la dimension temporelle à la plus petite taille trouvée
    pour pouvoir concaténer (si besoin).
    """
    agg = {
        "mains_reel": [],
        "mains_image": [],
        "deux_reel": [],
        "deux_image": []
    }
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
        X_list = []
        y_list = []
        for (X_c, y_c) in agg[cat]:
            # Tronquer si la longueur diffère
            X_list.append(X_c[..., :min_length])
            y_list.append(y_c)
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        out[cat] = (X_all, y_all)
        print(f"[INFO] Cat={cat}: total epochs={X_all.shape[0]}, time={min_length}")

    return out


############################################################################
# 7) Construction du pipeline FBCSP + LDA et exécution
############################################################################

if __name__ == "__main__":
    # On peut choisir le backend matplotlib
    matplotlib.use("TkAgg")  # ou 'Qt5Agg', 'Agg', etc.

    eeg_dir = os.getenv("EEG_DIR", "")
    print(f"[INFO] EEG_DIR={eeg_dir}")

    # Petit test/affichage sur S001R01
    test_subj = "S001"
    test_run = "S001R01.edf"
    path_example = os.path.join(eeg_dir, test_subj, test_run)
    raw_example = mne.io.read_raw_edf(path_example, preload=True)
    print("[INFO] Exemple brut:", raw_example.info)

    data_ex, times_ex = raw_example[:]
    fig_time, fig_fft = display_eeg_signals_and_spectra(
        data=data_ex,
        times=times_ex,
        sfreq=raw_example.info['sfreq'],
        title_suffix="-Brut",
        freq_xlim=(0, 80),
        fft_ylim=None,
        grid_shape=(8, 8)
    )
    plt.show()

    # Sélection de canaux : ex. on garde tout canal qui contient 'C', 'CP', ou 'FC'
    # possible_keywords = ['C', 'CP', 'FC']
    # possible_keywords = ['C']
    # chosen_channels = ['C3..', 'Cz..', 'C4..',] #0.615
    chosen_channels = ['C3..', 'Cz..', 'C4..', 'C1..', 'C2..', 'Fcz.', 'Fc3.', 'Fc4.', 'Cpz.', 'Cp3.', 'Cp4.', 'Cp1.', 'Cp2.'] #0.615
    # for ch in raw_example.info['ch_names']:
    #     if any(k in ch.upper() for k in possible_keywords):
    #         chosen_channels.append(ch)

    print(f"[INFO] Nombre de canaux retenus: {len(chosen_channels)} => {chosen_channels}")

    # Listing des sujets dans eeg_dir
    subject_dirs = list_subject_dirs(eeg_dir)
    print(f"[INFO] {len(subject_dirs)} sujets trouvés.")

    # Train / Test / Holdout
    train_dirs, tmp_dirs = train_test_split(subject_dirs, test_size=0.4, random_state=42)
    test_dirs, holdout_dirs = train_test_split(tmp_dirs, test_size=0.5, random_state=42)

    print(f"[INFO] Train: {train_dirs}")
    print(f"[INFO] Test: {test_dirs}")
    print(f"[INFO] Holdout: {holdout_dirs}")

    # Agrégation
    agg_train = aggregate_subjects(
        train_dirs, chosen_channels,
        l_freq=1.0,
        h_freq=40.0,
        tmin=0.5,
        tmax=3.8)
    agg_test = aggregate_subjects(
        test_dirs,
        chosen_channels,
        l_freq=1.0,
        h_freq=40.0,
        tmin=0.5,
        tmax=3.8)
    agg_hold = aggregate_subjects(
        holdout_dirs,
        chosen_channels,
        l_freq=1.0,
        h_freq=40.0,
        tmin=0.5,
        tmax=3.8)

    # Paramètres de nos sous-bandes (6 bandes)
    # filter_bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)] donne 0.62
    filter_bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)] #test sur 3 Cx.. channels = 0.6
    # filter_bands = [(4,8),(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)] #test sur 3 Cx.. channels = 0.6
    # filter_bands = [(8, 12), (16,24)] #test sur 3 Cx.. channels = 0.54

    def build_pipeline_fbcsp():
        """
        Construit le pipeline FilterBankCSP + LDA.
        On utilise 6 sous-bandes, 4 composantes CSP chacune.
        """
        fbcsp = FilterBankCSP(
            filter_bands=filter_bands,
            n_components=4,
            csp_reg='ledoit_wolf',
             csp_log=True,
            sfreq=160.0,
            n_jobs=16  # Utilise 16 threads pour le filtrage
        )
        clf = LDA(solver='lsqr', shrinkage='auto')
        return Pipeline([
            ('FBCSP', fbcsp),
            ('LDA', clf)
        ])

    categories = ["mains_reel", "mains_image", "deux_reel", "deux_image"]
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

        # Vérification : si on n'a qu'une classe, on skip
        if len(np.unique(y_train)) < 2:
            print(f"[WARN] Catégorie {cat} n'a qu'une seule classe en train, skip.")
            continue

        pipeline = build_pipeline_fbcsp()

        # Validation croisée sur train
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            print(f"[INFO] CV-scores={cv_scores}, mean={cv_scores.mean():.2f}")
        except Exception as e:
            print(f"[ERROR] Erreur cross_val pour {cat}: {e}")
            continue

        # Entraînement final + test
        pipeline.fit(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        print(f"[INFO] Test accuracy={test_acc:.2f}")

        # Sauvegarde du modèle
        models[cat] = pipeline
        model_filename = f"model_{cat}.pkl"
        with open(model_filename, "wb") as ff:
            pickle.dump(pipeline, ff)
        print(f"[INFO] Modèle sauvegardé => {model_filename}")

    holdout_accuracies = []
    # Évaluation sur le holdout
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

        # Afficher quelques prédictions
        for i in range(min(10, len(preds))):
            print(f"Epoch {i} => pred={preds[i]}, true={y_hold[i]}")

    # Calcul et affichage de la moyenne globale des holdout accuracies
    if holdout_accuracies:
        moyenne = np.mean(holdout_accuracies)
        print(f"\n[INFO] Moyenne holdout accuracy des 4 modèles : {moyenne:.3f}")