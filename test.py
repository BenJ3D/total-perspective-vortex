import os
import re

############################################################################
# Réglages de parallélisation BLAS / OpenMP
# Sur un AMD 5900X (12 coeurs / 24 threads), vous pouvez tester 12, 16, etc.
############################################################################
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

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


############################################################################
# 0) FilterBank CSP avec conversion en float64 pour éviter l'erreur "data copying"
############################################################################

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Implémentation simple d'un FilterBank CSP :
      - On applique plusieurs sous-bandes,
      - On entraîne un CSP par bande,
      - On concatène les features.

    On convertit explicitement les données en float64 pour éviter
    la ValueError "data copying was not requested..."
    """

    def __init__(self,
                 filter_bands=[(8,12), (12,16), (16,20), (20,24), (24,28), (28,32)],
                 n_components=4,
                 csp_reg='ledoit_wolf',
                 csp_log=True,
                 sfreq=160.0,
                 filter_length='auto',
                 n_jobs=1):
        self.filter_bands = filter_bands
        self.n_components = n_components
        self.csp_reg = csp_reg
        self.csp_log = csp_log
        self.sfreq = sfreq
        self.filter_length = filter_length
        self.n_jobs = n_jobs
        self._csps = []

    def fit(self, X, y):
        import mne.filter
        self._csps = []

        for (l_freq, h_freq) in self.filter_bands:
            # Filtrage sous-bande
            X_filt = mne.filter.filter_data(
                data=X,
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                filter_length=self.filter_length,
                n_jobs=self.n_jobs,
                verbose=False
            )
            # Conversion explicite en float64
            X_filt = X_filt.astype(np.float64)

            # CSP entraîné sur cette sous-bande
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
        feats_list = []
        for (l_freq, h_freq, csp) in self._csps:
            X_filt = mne.filter.filter_data(
                data=X,
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                filter_length=self.filter_length,
                n_jobs=self.n_jobs,
                verbose=False
            )
            X_filt = X_filt.astype(np.float64)

            X_csp = csp.transform(X_filt)
            feats_list.append(X_csp)

        return np.concatenate(feats_list, axis=1)


############################################################################
# 1) Fonction d'affichage signaux + FFT (pour debug / illustration)
############################################################################

def display_eeg_signals_and_spectra(data, times, sfreq, title_suffix="",
                                    time_ylim=None, freq_xlim=(0, 100),
                                    fft_ylim=None, grid_shape=None,
                                    custom_margins=None):
    import math
    n_channels = data.shape[0]

    if grid_shape is None:
        n_rows = int(math.ceil(math.sqrt(n_channels)))
        n_cols = int(math.ceil(n_channels / n_rows))
    else:
        n_rows, n_cols = grid_shape

    fig_time, axs_time = plt.subplots(n_rows, n_cols,
                                      figsize=(5 * n_cols, 4 * n_rows))
    axs_time = np.array(axs_time).reshape(-1) if n_channels > 1 else [axs_time]

    fig_fft, axs_fft = plt.subplots(n_rows, n_cols,
                                    figsize=(5 * n_cols, 4 * n_rows))
    axs_fft = np.array(axs_fft).reshape(-1) if n_channels > 1 else [axs_fft]

    if custom_margins is not None:
        fig_time.subplots_adjust(**custom_margins)
        fig_fft.subplots_adjust(**custom_margins)
    else:
        fig_time.subplots_adjust(left=0.07, right=0.986, top=0.967,
                                 bottom=0.06, wspace=0.2, hspace=0.2)
        fig_fft.subplots_adjust(left=0.07, right=0.986, top=0.967,
                                bottom=0.06, wspace=0.2, hspace=0.2)

    freqs = np.fft.rfftfreq(len(times), d=1/sfreq)

    for i in range(n_channels):
        # Time domain
        axs_time[i].plot(times, data[i])
        axs_time[i].set_title(f"Canal {i} {title_suffix}")
        axs_time[i].grid(True)
        axs_time[i].set_xlabel("Temps (s)")
        axs_time[i].set_ylabel("Amplitude")
        if time_ylim is not None:
            axs_time[i].set_ylim(time_ylim)

        # Freq domain
        fft_spectrum = np.abs(np.fft.rfft(data[i]))
        axs_fft[i].plot(freqs, fft_spectrum)
        axs_fft[i].set_title(f"FFT Canal {i} {title_suffix}")
        axs_fft[i].grid(True)
        axs_fft[i].set_xlabel("Fréquence (Hz)")
        axs_fft[i].set_ylabel("Amplitude")
        axs_fft[i].set_xlim(freq_xlim)
        if fft_ylim is not None:
            axs_fft[i].set_ylim(fft_ylim)

    plt.tight_layout()
    return fig_time, fig_fft


############################################################################
# 2) get_run_category_4 : renvoie 4 catégories (mains_reel, mains_image, deux_reel, deux_image)
############################################################################

def get_run_category_4(file_name):
    """
    On se base sur les runs :
      - mains_reel   = runs  3,  7, 11
      - mains_image  = runs  4,  8, 12
      - deux_reel    = runs  5,  9, 13
      - deux_image   = runs  6, 10, 14
    """
    m_obj = re.search(r'R(\d+)', file_name)
    if not m_obj:
        return None
    run = int(m_obj.group(1))

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
# 3) process_edf et process_subject
############################################################################

def process_edf(file_path,
                channels_to_keep,
                l_freq=1.0, h_freq=40.0,
                tmin=0.5, tmax=3.0,
                do_reref=True):
    """Charge un .edf, sélectionne canaux, filtre, segmente, renvoie un mne.Epochs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Limited 1 annotation")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # On ne garde que les canaux désirés
    raw.pick(channels_to_keep)

    # Re-référence EEG globale
    if do_reref:
        raw.set_eeg_reference('average', projection=False, verbose=False)

    # Filtrage [l_freq, h_freq]
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin',
               verbose=False, n_jobs=1)

    # On récupère les events T1/T2
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    selected_event_id = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
    if len(selected_event_id) == 0:
        return None

    # Construction epochs
    epochs = mne.Epochs(
        raw, events, event_id=selected_event_id,
        tmin=tmin, tmax=tmax, baseline=None,
        preload=True, verbose=False, on_missing='ignore'
    )

    # Vérification du nb d'échantillons
    sfreq = raw.info['sfreq']
    expected_len = int(round((tmax - tmin) * sfreq)) + 1
    data = epochs.get_data()  # shape = (n_epochs, n_channels, n_times)
    good_idx = [i for i in range(data.shape[0]) if data[i].shape[-1] == expected_len]
    if len(good_idx) < data.shape[0]:
        if len(good_idx) == 0:
            return None
        epochs = epochs[good_idx]

    return epochs


def process_subject(subject_dir, channels,
                    l_freq=1.0, h_freq=40.0,
                    tmin=0.5, tmax=3.0):
    """
    Parcourt tous les .edf d'un sujet, les regroupe par catégorie (4 catégories).
    Renvoie un dict: { 'mains_reel': (X, y), 'mains_image': (X, y), 'deux_reel': ..., 'deux_image': ... }
    """
    cat_list = ["mains_reel", "mains_image", "deux_reel", "deux_image"]
    cat_epochs = {c: [] for c in cat_list}

    for filename in os.listdir(subject_dir):
        if not filename.endswith(".edf"):
            continue
        cat = get_run_category_4(filename)
        if cat is None:
            # Ce run n'appartient pas aux 4 cat (p.ex. run1, run2 => baseline)
            continue

        path = os.path.join(subject_dir, filename)
        epochs = process_edf(path, channels, l_freq, h_freq, tmin, tmax)
        if epochs is not None and len(epochs) > 0:
            cat_epochs[cat].append(epochs)
            print(f"[INFO] {filename}: {len(epochs)} epochs -> {cat}")
        else:
            print(f"[INFO] {filename}: 0 epoch => skip.")

    # Concatène tout par catégorie
    out = {}
    for cat in cat_list:
        if len(cat_epochs[cat]) == 0:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=RuntimeWarning,
                                    message=".*Concatenation of Annotations.*")
            try:
                ep_all = mne.concatenate_epochs(cat_epochs[cat])
            except ValueError:
                continue

        X = ep_all.get_data()          # (n_epochs, n_chan, n_times)
        y = ep_all.events[:, 2]        # T1=1 ou T2=2
        if X.size > 0:
            out[cat] = (X, y)
            print(f"[INFO] {subject_dir} => cat={cat}, total={X.shape[0]} epochs")
    return out if len(out) > 0 else None


############################################################################
# 4) Agrégation multi-sujets
############################################################################

def list_subject_dirs(eeg_dir):
    dirs = []
    for d in os.listdir(eeg_dir):
        p = os.path.join(eeg_dir, d)
        if os.path.isdir(p) and d.startswith("S") and len(d) == 4:
            dirs.append(p)
    dirs.sort()
    return dirs


def aggregate_subjects(subject_dirs,
                       channels,
                       l_freq=1.0, h_freq=40.0,
                       tmin=0.5, tmax=3.0):
    """
    Agrège (X, y) sur tous les sujets pour nos 4 catégories.
    out = {
       'mains_reel': (X, y),
       'mains_image': (X, y),
       'deux_reel': (X, y),
       'deux_image': (X, y)
    }
    """
    cat_list = ["mains_reel", "mains_image", "deux_reel", "deux_image"]
    agg = {c: [] for c in cat_list}

    for sd in subject_dirs:
        data_sub = process_subject(sd, channels, l_freq, h_freq, tmin, tmax)
        if data_sub is None:
            continue
        for cat in data_sub:
            Xc, yc = data_sub[cat]
            if Xc.size > 0:
                agg[cat].append((Xc, yc))

    out = {}
    for cat in cat_list:
        if len(agg[cat]) == 0:
            continue

        # On égalise la dimension temporelle
        min_len = min(x[0].shape[-1] for x in agg[cat])
        X_list = []
        y_list = []
        for (Xc, yc) in agg[cat]:
            X_list.append(Xc[..., :min_len])
            y_list.append(yc)

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        out[cat] = (X_all, y_all)
        print(f"[INFO] cat={cat}: total={X_all.shape[0]} epochs, time={min_len}")

    return out


############################################################################
# 5) Script principal
############################################################################

if __name__ == "__main__":
    matplotlib.use("TkAgg")

    eeg_dir = os.getenv("EEG_DIR", "")
    print("[INFO] EEG_DIR =", eeg_dir)

    # Juste pour montrer un exemple d'affichage
    sub_test = "S001"
    run_test = "S001R01.edf"
    path_test = os.path.join(eeg_dir, sub_test, run_test)
    raw_ex = mne.io.read_raw_edf(path_test, preload=True)
    print("[INFO] Example raw:", raw_ex.info)
    data_ex, times_ex = raw_ex[:]

    fig_t, fig_f = display_eeg_signals_and_spectra(
        data=data_ex,
        times=times_ex,
        sfreq=raw_ex.info["sfreq"],
        title_suffix="-Ex",
        freq_xlim=(0,80)
    )
    plt.show()

    # Sélection de canaux sensorimoteurs
    # Vous pouvez ajuster selon vos observations
    chosen_channels = [
        "C3..", "Cz..", "C4.."
    ]
    print(f"[INFO] chosen_channels={chosen_channels}, total={len(chosen_channels)}")

    # Liste des sujets
    all_subdirs = list_subject_dirs(eeg_dir)
    print("[INFO] Nombre de sujets =", len(all_subdirs))

    # Split train / test / holdout
    train_dirs, tmp_dirs = train_test_split(all_subdirs, test_size=0.4, random_state=42)
    test_dirs, holdout_dirs = train_test_split(tmp_dirs, test_size=0.5, random_state=42)

    print("[INFO] Train dirs:", train_dirs)
    print("[INFO] Test dirs :", test_dirs)
    print("[INFO] Holdout  :", holdout_dirs)

    # Agrégation
    agg_train = aggregate_subjects(train_dirs, chosen_channels,
                                   l_freq=1.0, h_freq=40.0,
                                   tmin=0.5, tmax=3.0)
    agg_test  = aggregate_subjects(test_dirs,  chosen_channels,
                                   l_freq=1.0, h_freq=40.0,
                                   tmin=0.5, tmax=3.0)
    agg_hold  = aggregate_subjects(holdout_dirs, chosen_channels,
                                   l_freq=1.0, h_freq=40.0,
                                   tmin=0.5, tmax=3.0)

    # Construit un pipeline FilterBank CSP + LDA
    def build_pipeline_fbcsp():
        fbcsp = FilterBankCSP(
            filter_bands=[(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)],
            # filter_bands=[(7,32)],
            n_components=4,
            csp_reg='ledoit_wolf',  # ou 'oas', ...
            csp_log=True,
            sfreq=160.0,
            filter_length='auto',
            n_jobs=1
        )
        clf = LDA(solver='lsqr', shrinkage='auto')
        return Pipeline([('FBCSP', fbcsp), ('LDA', clf)])

    categories = ["mains_reel", "mains_image", "deux_reel", "deux_image"]
    models = {}

    for cat in categories:
        if cat not in agg_train:
            print(f"[WARN] No train data for {cat} => skip.")
            continue
        if cat not in agg_test:
            print(f"[WARN] No test data for {cat} => skip.")
            continue

        Xtr, ytr = agg_train[cat]
        Xte, yte = agg_test[cat]
        print(f"\n[INFO] Cat={cat}, Xtr={Xtr.shape}, ytr={ytr.shape}, Xte={Xte.shape}")

        pipeline = build_pipeline_fbcsp()

        # Cross-validation sur le train
        cv_scores = cross_val_score(
            pipeline, Xtr, ytr,
            cv=5, scoring='accuracy', n_jobs=-1,
            error_score='raise'
        )
        print(f"[INFO] CV={cv_scores}, mean={cv_scores.mean():.2f}")

        pipeline.fit(Xtr, ytr)
        test_score = pipeline.score(Xte, yte)
        print(f"[INFO] Test accuracy={test_score:.2f}")

        models[cat] = pipeline
        model_filename = f"model_{cat}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"[INFO] Sauvegarde modèle => {model_filename}")

    # Évaluation sur le holdout
    for cat in models:
        if cat not in agg_hold:
            print(f"[INFO] cat={cat}, no holdout => skip.")
            continue

        Xh, yh = agg_hold[cat]
        hold_score = models[cat].score(Xh, yh)
        print(f"\n[INFO] HOLDOUT cat={cat}, shape={Xh.shape}, accuracy={hold_score:.2f}")

        preds = models[cat].predict(Xh[:10])
        print("First 10 predictions vs truth:")
        for i in range(10):
            print(f"   epoch={i}, pred={preds[i]}, true={yh[i]}")
