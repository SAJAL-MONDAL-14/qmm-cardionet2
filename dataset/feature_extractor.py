"""
feature_extractor.py  —  Research-Grade ECG Feature Extraction
===============================================================
65 features per lead across 10 groups:
  1. Time-domain stats (10)     6. Morphology (6)
  2. R-peak / RR (8)            7. Wavelet energy (5)
  3. HRV time-domain (10)       8. Spectral features (8)
  4. HRV frequency-domain (6)   9. Hjorth parameters (3)
  5. HRV non-linear (5)        10. Zero-crossing (4)

Multi-lead: extract_ecg_features_multilead() → 5 leads × 65 = 325 features
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

FS = 500   # PTB-XL high-res sampling frequency


# ── Utilities ─────────────────────────────────────────────

def _bandpass_filter(sig, lowcut=0.5, highcut=45.0, fs=FS, order=4):
    nyq  = fs / 2.0
    b, a = sp_signal.butter(order, [lowcut/nyq, min(highcut/nyq, 0.99)], btype="band")
    return sp_signal.filtfilt(b, a, sig)

def _detect_r_peaks(sig, fs=FS):
    filtered   = _bandpass_filter(sig, 5, 15, fs)
    diff       = np.diff(filtered) ** 2
    win        = int(0.15 * fs)
    integrated = np.convolve(diff, np.ones(win)/win, mode="same")
    threshold  = 0.3 * np.max(integrated)
    peaks, _   = sp_signal.find_peaks(integrated, height=threshold,
                                       distance=int(0.2*fs))
    return peaks

def _rr_intervals(r_peaks, fs=FS):
    return np.diff(r_peaks) / fs * 1000.0 if len(r_peaks) >= 2 else np.array([])

def _safe(arr, func, fallback=0.0):
    try:
        return float(func(arr)) if len(arr) >= 2 else fallback
    except Exception:
        return fallback


# ── Feature Groups ────────────────────────────────────────

def _time_domain_stats(sig):
    return [np.mean(sig), np.std(sig), np.var(sig),
            np.max(sig), np.min(sig), np.max(sig)-np.min(sig),
            np.median(sig), float(skew(sig)), float(kurtosis(sig)),
            np.sqrt(np.mean(sig**2))]

def _rr_basic(r_peaks, fs=FS):
    rr = _rr_intervals(r_peaks, fs)
    hr = 60000.0 / (_safe(rr, np.mean, 1000)) if len(rr) > 0 else 0
    return [float(len(r_peaks)), hr,
            _safe(rr, np.mean), _safe(rr, np.std),
            _safe(rr, np.min),  _safe(rr, np.max),
            _safe(rr, lambda x: np.max(x)-np.min(x)),
            _safe(rr, np.median)]

def _hrv_time(r_peaks, fs=FS):
    rr = _rr_intervals(r_peaks, fs)
    if len(rr) < 3:
        return [0.0] * 10
    d  = np.diff(rr)
    return [np.std(rr),
            np.sqrt(np.mean(d**2)),
            np.std(d),
            float(np.sum(np.abs(d) > 50)),
            float(np.sum(np.abs(d) > 50)) / len(d) * 100,
            float(np.sum(np.abs(d) > 20)),
            float(np.sum(np.abs(d) > 20)) / len(d) * 100,
            np.std(rr) / (np.mean(rr)+1e-8),
            np.std(rr) / (np.mean(rr)+1e-8),
            np.mean(np.abs(d))]

def _hrv_freq(r_peaks, fs=FS):
    rr = _rr_intervals(r_peaks, fs)
    if len(rr) < 10:
        return [0.0] * 6
    fs_rr  = 4.0
    t_rr   = np.cumsum(rr) / 1000.0
    t_even = np.arange(t_rr[0], t_rr[-1], 1.0/fs_rr)
    if len(t_even) < 8:
        return [0.0] * 6
    rr_e   = np.interp(t_even, t_rr, rr)
    f, psd = sp_signal.welch(rr_e, fs=fs_rr, nperseg=min(len(rr_e), 64))
    def bp(lo, hi):
        idx = (f >= lo) & (f < hi)
        return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0
    vlf = bp(0.003, 0.04); lf = bp(0.04, 0.15); hf = bp(0.15, 0.40)
    tot = vlf + lf + hf + 1e-8
    return [vlf, lf, hf, lf/tot, hf/tot, lf/(hf+1e-8)]

def _hrv_nonlinear(r_peaks, fs=FS):
    rr = _rr_intervals(r_peaks, fs)
    if len(rr) < 4:
        return [0.0] * 5
    d   = np.diff(rr)
    sd1 = np.std(d) / np.sqrt(2)
    sd2 = np.sqrt(max(2*np.std(rr)**2 - sd1**2, 0))
    def sampen(x, m=2):
        r = 0.2 * np.std(x); n = len(x)
        def cnt(m):
            c = 0
            for i in range(n-m):
                for j in range(i+1, n-m):
                    if np.max(np.abs(x[i:i+m]-x[j:j+m])) < r:
                        c += 1
            return c
        try:
            B = cnt(m); A = cnt(m+1)
            return -np.log((A+1e-8)/(B+1e-8))
        except Exception:
            return 0.0
    rr_s = rr[:50] if len(rr) > 50 else rr
    return [sd1, sd2, sd1/(sd2+1e-8), np.pi*sd1*sd2, sampen(rr_s)]

def _morphology(sig, r_peaks, fs=FS):
    if len(r_peaks) < 2:
        return [0.0] * 6
    wq = int(0.06*fs); wt = int(0.20*fs); st = int(0.08*fs)
    qd, qa, ta, stl, pr, qt = [], [], [], [], [], []
    for r in r_peaks:
        seg = sig[max(0,r-wq):min(len(sig),r+wq)]
        if len(seg) > 0:
            qa.append(np.max(seg)-np.min(seg))
            above = np.where(seg > 0.5*sig[r])[0]
            qd.append(len(above)/fs*1000 if len(above) > 0 else 0)
        t0 = min(r+st, len(sig)); t1 = min(r+wt, len(sig))
        if t1 > t0:
            ta.append(np.max(sig[t0:t1]))
        stl.append(sig[min(r+st, len(sig)-1)])
        lo = max(0, r-int(0.25*fs)); seg2 = sig[lo:r]
        zc = np.where(np.diff(np.sign(seg2)))[0]
        if len(zc) > 0:
            pr.append((r-(lo+zc[-1]))/fs*1000)
        tlo = min(r+st, len(sig)); thi = min(r+int(0.45*fs), len(sig))
        if thi > tlo:
            qt.append((tlo+np.argmax(sig[tlo:thi])-r)/fs*1000)
    return [np.mean(qd) if qd else 0, np.mean(qa) if qa else 0,
            np.mean(ta) if ta else 0, np.mean(stl) if stl else 0,
            np.mean(pr) if pr else 0, np.mean(qt) if qt else 0]

def _wavelet_energy(sig):
    energies = []; x = sig.copy().astype(np.float64)
    for _ in range(5):
        if len(x) < 2:
            energies.append(0.0); continue
        detail = (x[:-1:2] - x[1::2]) / np.sqrt(2)
        energies.append(float(np.sum(detail**2)))
        x = (x[:-1:2] + x[1::2]) / np.sqrt(2)
    return energies

def _spectral(sig, fs=FS):
    f, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 1024))
    def bp(lo, hi):
        idx = (f >= lo) & (f < hi)
        return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0
    d=bp(0.5,3); th=bp(3,8); al=bp(8,13); be=bp(13,30); ga=bp(30,45)
    tot = d+th+al+be+ga+1e-8
    pn  = psd/(psd.sum()+1e-8)
    return [d/tot, th/tot, al/tot, be/tot, ga/tot,
            -float(np.sum(pn*np.log(pn+1e-8))),
            float(f[np.argmax(psd)]),
            float(np.sum(f*psd)/(np.sum(psd)+1e-8))]

def _hjorth(sig):
    act = np.var(sig); d1 = np.diff(sig); d2 = np.diff(d1)
    mob = np.sqrt(np.var(d1)/(act+1e-8))
    com = np.sqrt(np.var(d2)/(np.var(d1)+1e-8))/(mob+1e-8)
    return [float(act), float(mob), float(com)]

def _zc_features(sig, fs=FS):
    zc  = np.sum(np.diff(np.sign(sig)) != 0)/(len(sig)/fs)
    mad = np.mean(np.abs(sig-np.mean(sig)))
    en  = np.sum(sig**2)/len(sig)
    ll  = np.sum(np.abs(np.diff(sig)))
    return [float(zc), float(mad), float(en), float(ll)]


# ── Public API ────────────────────────────────────────────

def extract_ecg_features(sig, fs=FS):
    """Extract 65 features from a single ECG lead."""
    sig = np.nan_to_num(np.array(sig, dtype=np.float64), nan=0.0)
    flt = _bandpass_filter(sig, fs=fs)
    try:
        r_peaks = _detect_r_peaks(flt, fs)
    except Exception:
        r_peaks = np.array([])

    feats = []
    feats.extend(_time_domain_stats(flt))
    feats.extend(_rr_basic(r_peaks, fs))
    feats.extend(_hrv_time(r_peaks, fs))
    feats.extend(_hrv_freq(r_peaks, fs))
    feats.extend(_hrv_nonlinear(r_peaks, fs))
    feats.extend(_morphology(flt, r_peaks, fs))
    feats.extend(_wavelet_energy(flt))
    feats.extend(_spectral(flt, fs))
    feats.extend(_hjorth(flt))
    feats.extend(_zc_features(flt, fs))

    out = np.array(feats, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


SELECTED_LEADS = [0, 1, 6, 7, 10]   # Lead I, II, V1, V2, V5

def extract_ecg_features_multilead(signal_matrix, fs=FS, leads=SELECTED_LEADS):
    """Extract features from 5 leads → 325 features total."""
    return np.concatenate([
        extract_ecg_features(signal_matrix[:, l].astype(np.float64), fs)
        for l in leads
    ])

def get_feature_names(multilead=False, leads=SELECTED_LEADS):
    base = [
        "mean","std","variance","max","min","peak_to_peak","median",
        "skewness","kurtosis","rms",
        "n_r_peaks","mean_hr_bpm","mean_rr_ms","std_rr_ms","min_rr_ms",
        "max_rr_ms","rr_range_ms","median_rr_ms",
        "sdnn","rmssd","sdsd","nn50","pnn50","nn20","pnn20","cvnn",
        "cv_rr","mean_abs_diff_rr",
        "vlf_power","lf_power","hf_power","lf_norm","hf_norm","lf_hf_ratio",
        "sd1","sd2","sd_ratio","poincare_area","sample_entropy",
        "qrs_duration_ms","qrs_amplitude","t_wave_amplitude","st_level",
        "pr_interval_ms","qt_interval_ms",
        "wavelet_d1","wavelet_d2","wavelet_d3","wavelet_d4","wavelet_d5",
        "delta_ratio","theta_ratio","alpha_ratio","beta_ratio","gamma_ratio",
        "spectral_entropy","dominant_freq","spectral_centroid",
        "hjorth_activity","hjorth_mobility","hjorth_complexity",
        "zero_crossing_rate","mean_abs_deviation","signal_energy","line_length",
    ]
    if not multilead:
        return [f"leadI_{n}" for n in base]
    lnames = ["I","II","V1","V2","V3","V4","V5","V6","aVR","aVL","aVF","V5b"]
    names  = []
    for li in leads:
        ln = lnames[li] if li < len(lnames) else f"L{li}"
        names.extend([f"{ln}_{n}" for n in base])
    return names


if __name__ == "__main__":
    dummy = np.sin(2*np.pi*1.2*np.arange(5000)/500) + np.random.randn(5000)*0.05
    f = extract_ecg_features(dummy)
    print(f"Single-lead features: {f.shape[0]}  NaN={np.isnan(f).any()}")
    ml = extract_ecg_features_multilead(np.random.randn(5000, 12))
    print(f"Multi-lead features : {ml.shape[0]}")