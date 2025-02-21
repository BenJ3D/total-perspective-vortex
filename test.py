import os
import re
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import warnings

from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

# Constante minimale pour la longueur du signal pour le filtrage
MIN_SIGNAL_LENGTH = 265


############################################################################
# 1) Classe FilterBankCSP : applique plusieurs bandes de fréquences, CSP, puis concatène
############################################################################

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Implémentation du Filter Bank CSP.
    Pour chaque sous-bande (définie dans filter_bands), on entraîne un CSP séparé.
    Lors du transform, on filtre, on applique le CSP et on concatène les features.

    X est supposé avoir la forme (n_trials, n_channels, n_times)
    """

    def __init__(self,
                 filter_bands=[(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)],
                 n_components=4,
                 csp_reg='ledoit_wolf',
                 csp_log=True,
                 sfreq=160.0):
        self.filter_bands = filter_bands
        self.n_components = n_components
        self.csp_reg = csp_reg
        self.csp_log = csp_log
        self.sfreq = sfreq
        self._csps = []  # liste des objets CSP par bande

    def _pad_if_needed(self, X):
        n_times = X.shape[-1]
        if n_times < MIN_SIGNAL_LENGTH:
            pad_width = MIN_SIGNAL_LENGTH - n_times
            X_padded = np.pad(X, pad_width=((0, 0), (0, 0), (0, pad_width)), mode='constant')
            return X_padded, n_times  # on retournera plus tard le signal original (non‑padé) en recadrant
        else:
            return X, None

    def fit(self, X, y):
        import mne.filter  # pour filter_data
        self._csps = []
        # Pad X si nécessaire
        X_use, orig_length = self._pad_if_needed(X)
        for (l_freq, h_freq) in self.filter_bands:
            # Filtrage dans la bande [l_freq, h_freq]
            X_filt_full = mne.filter.filter_data(
                data=X_use, sfreq=self.sfreq,
                l_freq=l_freq, h_freq=h_freq,
                verbose=False
            )
            # Si le signal avait été padé, recadrer à la longueur d'origine
            if orig_length is not None:
                X_filt = X_filt_full[..., :orig_length]
            else:
                X_filt = X_filt_full
            # Création et entraînement du CSP
            csp = CSP(n_components=self.n_components,
                      reg=self.csp_reg,
                      norm_trace=False,
                      log=self.csp_log)
            csp.fit(X_filt, y)
            self._csps.append((l_freq, h_freq, csp))
        return self

    def transform(self, X):
        import mne.filter
        X_use, orig_length = self._pad_if_needed(X)
        X_features_list = []
        for (l_freq, h_freq, csp) in self._csps:
            X_filt_full = mne.filter.filter_data(
                data=X_use, sfreq=self.sfreq,
                l_freq=l_freq, h_freq=h_freq,
                verbose=False
            )
            if orig_length is not None:
                X_filt = X_filt_full[..., :orig_length]
            else:
                X_filt = X_filt_full
            X_csp = csp.transform(X_filt)  # (n_epochs, n_components)
            X_features_list.append(X_csp)
        return np.concatenate(X_features_list, axis=1)  # (n_epochs, total_components)


############################################################################
# 2) Fonctions d'affichage
############################################################################

def display_eeg_signals_and_spectra(data, times, sfreq, title_suffix="",
                                    time_ylim=None, freq_xlim=(0, 100),
                                    fft_ylim=None, grid_shape=None,
                                    custom_margins=None):
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
# 3) Gestion des runs et détermination de catégorie
############################################################################

def get_run_category(file_name):
    """
    Détermine la catégorie à partir du numéro de run.
      - Renvoie None pour les runs de baseline (<=2)
      - 'mains_reel' pour runs [3, 7, 11]
      - 'mains_image' pour runs [4, 8, 12]
      - 'deux_reel' pour runs [5, 9, 13]
      - 'deux_image' pour runs [6, 10, 14]
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

def process_edf(file_path, channels_to_keep, l_freq, h_freq,
                tmin=0.5, tmax=2.5, do_reref=True):
    """
    Charge le fichier EDF, conserve uniquement les canaux souhaités, (optionnellement)
    re‑référence, filtre dans [l_freq, h_freq] et découpe en epochs sur [tmin, tmax].
    Les epochs dont la taille n’est pas conforme sont éliminées.
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
    epochs = mne.Epochs(raw, events, event_id=selected_event_id,
                        tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, verbose=False, on_missing='ignore')
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
# 5) Traitement d'un sujet : ne conserve pas la séparation mains en gauche/droite
#    (pour garantir deux classes par catégorie)
############################################################################

def process_subject(subject_dir, channels_to_keep, l_freq, h_freq, tmin, tmax):
    """
    Parcourt tous les fichiers .edf dans subject_dir, détermine la catégorie via get_run_category,
    et découpe en epochs. Pour les tâches "mains", on conserve l’ensemble (pour avoir à la fois T1 et T2).
    Renvoie un dictionnaire : { 'mains_reel': (X,y), 'mains_image': (X,y),
                                'deux_reel': (X,y), 'deux_image': (X,y) }.
    """
    cat_epochs = {
        "mains_reel": [],
        "mains_image": [],
        "deux_reel": [],
        "deux_image": []
    }
    for file in os.listdir(subject_dir):
        if file.endswith(".edf"):
            cat = get_run_category(file)
            if cat is not None:
                file_path = os.path.join(subject_dir, file)
                epochs = process_edf(file_path, channels_to_keep, l_freq, h_freq, tmin, tmax, do_reref=True)
                if epochs is not None and len(epochs) > 0:
                    cat_epochs[cat].append(epochs)
                    print(f"[INFO] {file}: {len(epochs)} epochs -> {cat}")
                else:
                    print(f"[INFO] {file}: 0 epoch -> ignoré.")
    out = {}
    for cat in cat_epochs:
        if len(cat_epochs[cat]) > 0:
            try:
                epochs_all = mne.concatenate_epochs(cat_epochs[cat])
            except ValueError:
                print(f"[INFO] Aucune epoch valide pour cat={cat} dans {subject_dir}")
                continue
            X = epochs_all.get_data()  # (n_epochs, n_channels, n_times)
            y = epochs_all.events[:, 2]
            if X.size > 0:
                out[cat] = (X, y)
                print(f"[INFO] Subject {os.path.basename(subject_dir)} - {cat}: {X.shape[0]} epochs")
    return out if len(out) > 0 else None


############################################################################
# 6) Agrégation multi-sujets
############################################################################

def list_subject_dirs(eeg_dir):
    dirs = []
    for d in os.listdir(eeg_dir):
        path = os.path.join(eeg_dir, d)
        if os.path.isdir(path) and d.startswith("S") and len(d) == 4:
            dirs.append(path)
    dirs.sort()
    return dirs


def aggregate_subjects(subject_dirs, channels, l_freq, h_freq, tmin, tmax):
    """
    Agrège pour chaque catégorie (les 4 classes) les données de tous les sujets.
    Pour chaque catégorie, on tronque la dimension temporelle à la plus courte avant concaténation.
    Renvoie un dictionnaire : cat -> (X, y)
    """
    agg = {
        "mains_reel": [],
        "mains_image": [],
        "deux_reel": [],
        "deux_image": []
    }
    for sd in subject_dirs:
        subj_data = process_subject(sd, channels, l_freq, h_freq, tmin, tmax)
        if subj_data is not None:
            for cat in subj_data:
                if subj_data[cat][0].size > 0:
                    agg[cat].append(subj_data[cat])
    out = {}
    for cat in agg:
        if len(agg[cat]) == 0:
            continue
        min_length = min(x[0].shape[-1] for x in agg[cat])
        X_list = []
        y_list = []
        for (X_c, y_c) in agg[cat]:
            X_list.append(X_c[..., :min_length])  # tronquer à la longueur minimale
            y_list.append(y_c)
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        out[cat] = (X_all, y_all)
        print(f"[INFO] Cat={cat}: total epochs={X_all.shape[0]}, time={min_length}")
    return out


############################################################################
# 7) Construction du pipeline et exécution globale
############################################################################

if __name__ == "__main__":
    matplotlib.use("TkAgg")  # ou 'Qt5Agg', etc.

    # Récupération du répertoire EEG via la variable d'environnement EEG_DIR
    eeg_dir = os.getenv("EEG_DIR", "")
    print(f"[INFO] EEG_DIR={eeg_dir}")

    # Exemple d'affichage d'un sujet pour inspection
    test_subj = "S001"
    test_run = "S001R01.edf"
    file_example = os.path.join(eeg_dir, test_subj, test_run)
    raw_example = mne.io.read_raw_edf(file_example, preload=True)
    print("[INFO] Exemple brut:", raw_example.info)
    data_ex, times_ex = raw_example[:]
    fig_time, fig_fft = display_eeg_signals_and_spectra(
        data=data_ex,
        times=times_ex,
        sfreq=raw_example.info['sfreq'],
        title_suffix="-Brut",
        freq_xlim=(0, 80),
        fft_ylim=(0, 0.03),
        grid_shape=(8, 8)
    )
    plt.show()

    # Sélection "large" de canaux sensorimoteurs : on retient tous les canaux dont le nom contient 'C', 'CP' ou 'FC'
    possible_keywords = ['C', 'CP', 'FC']
    chosen_channels = []
    for ch in raw_example.info['ch_names']:
        if any(k in ch.upper() for k in possible_keywords):
            chosen_channels.append(ch)
    print(f"[INFO] Nombre de canaux retenus: {len(chosen_channels)} => {chosen_channels}")

    # Liste des dossiers sujets
    subject_dirs = list_subject_dirs(eeg_dir)
    print(f"[INFO] {len(subject_dirs)} sujets trouvés.")

    # Séparation train / test / holdout
    train_dirs, temp_dirs = train_test_split(subject_dirs, test_size=0.4, random_state=42)
    test_dirs, holdout_dirs = train_test_split(temp_dirs, test_size=0.5, random_state=42)
    print(f"[INFO] Train: {train_dirs}")
    print(f"[INFO] Test: {test_dirs}")
    print(f"[INFO] Holdout: {holdout_dirs}")

    # Agrégation des données (fenêtre tmin=0.5, tmax=2.5 => environ 2 s)
    agg_train = aggregate_subjects(train_dirs, chosen_channels, l_freq=1.0, h_freq=40.0, tmin=0.5, tmax=2.5)
    agg_test = aggregate_subjects(test_dirs, chosen_channels, l_freq=1.0, h_freq=40.0, tmin=0.5, tmax=2.5)
    agg_holdout = aggregate_subjects(holdout_dirs, chosen_channels, l_freq=1.0, h_freq=40.0, tmin=0.5, tmax=2.5)

    # Construction d'un pipeline FBCSP + LDA
    # On utilise ici 6 sous-bandes dans le FilterBankCSP et 4 composantes par sous-bande.
    filter_bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]


    def build_pipeline_fbcsp():
        fbcsp = FilterBankCSP(
            filter_bands=filter_bands,
            n_components=4,
            csp_reg='ledoit_wolf',
            csp_log=True,
            sfreq=160.0
        )
        clf = LDA(solver='lsqr', shrinkage='auto')
        return Pipeline([('FBCSP', fbcsp), ('LDA', clf)])


    # On entraîne un modèle pour chacune des 4 catégories
    models = {}
    categories = ["mains_reel", "mains_image", "deux_reel", "deux_image"]
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
        # Si le jeu d'entraînement ne contient qu'une seule classe, on passe cette catégorie.
        if len(np.unique(y_train)) < 2:
            print(f"[WARN] Catégorie {cat} n'a qu'une seule classe dans l'entraînement, on skip cette catégorie.")
            continue

        pipeline = build_pipeline_fbcsp()
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            print(f"[INFO] CV-scores={cv_scores}, mean={cv_scores.mean():.2f}")
        except Exception as e:
            print(f"[ERROR] Erreur en cross validation pour {cat}: {e}")
            continue

        pipeline.fit(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        print(f"[INFO] Test accuracy={test_score:.2f}")

        models[cat] = pipeline
        with open(f"model_{cat}.pkl", "wb") as ff:
            pickle.dump(pipeline, ff)
        print(f"[INFO] Sauvegardé => model_{cat}.pkl")

    # Prédiction sur le jeu holdout pour chaque catégorie entraînée
    for cat in models:
        if cat not in agg_holdout:
            print(f"[INFO] cat={cat}, pas de data holdout => skip.")
            continue
        X_hold, y_hold = agg_holdout[cat]
        print(f"\n[INFO] HOLDOUT prédictions cat={cat}, shape={X_hold.shape}")
        pipeline = models[cat]
        preds = pipeline.predict(X_hold)
        hold_acc = np.mean(preds == y_hold)
        print(f"[INFO] Holdout accuracy={hold_acc:.2f}")
        for i in range(min(10, len(preds))):
            print(f"Epoch {i} => pred={preds[i]}, true={y_hold[i]}")
