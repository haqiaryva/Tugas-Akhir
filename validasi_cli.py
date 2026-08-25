import pandas as pd
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

FLOODS_RAW = {
    2022: [
        'Mandalajati', 'Bandung Wetan', 'Bojongloa Kaler',
        'Cimahi Utara', 'Cimahi Tengah', 'Cimahi Selatan',
        'Arjasari', 'Baleendah', 'Bojongsoang', 'Cangkuang', 'Cicalengka',
        'Cileunyi', 'Ciparay', 'Ciwidey', 'Dayeuhkolot', 'Katapang',
        'Margaasih', 'Nagreg', 'Pameungpeuk', 'Pangalengan', 'Pasirjambu',
        'Rancaekek', 'Solokanjeruk', 'Soreang', 'Ngamprah', 'Padalarang', 'Parongpong',
    ],
    2023: [
        'Bandung Kulon', 'Sukasari',
        'Cimahi Selatan', 'Cimahi Tengah', 'Cimahi Utara',
        'Arjasari', 'Baleendah', 'Banjaran', 'Bojongsoang', 'Cangkuang', 'Cicalengka',
        'Cimenyan', 'Ciparay', 'Ciwidey', 'Dayeuhkolot', 'Ibun', 'Katapang',
        'Kertasari', 'Kutawaringin', 'Majalaya', 'Margaasih', 'Pameungpeuk',
        'Pangalengan', 'Paseh', 'Rancaekek', 'Soreang',
        'Cihampelas', 'Cisarua', 'Padalarang', 'Sindangkerta',
    ],
    2024: [
        'Antapani', 'Arcamanik', 'Bandung Kidul', 'Batununggal', 'Buahbatu',
        'Cibeunying Kaler', 'Cibeunying Kidul', 'Gedebage', 'Mandalajati',
        'Panyileukan', 'Sumur Bandung',
        'Cimahi Tengah', 'Cimahi Utara',   # Cimahi Selatan: 0 kejadian
        'Arjasari', 'Baleendah', 'Banjaran', 'Bojongsoang', 'Cangkuang', 'Cicalengka',
        'Ciparay', 'Ciwidey', 'Dayeuhkolot', 'Ibun', 'Katapang', 'Kertasari',
        'Majalaya', 'Nagreg', 'Pacet', 'Pameungpeuk', 'Pangalengan', 'Paseh',
        'Pasirjambu', 'Rancaekek', 'Solokanjeruk', 'Soreang',
        'Cikalongwetan', 'Cipatat', 'Cipongkor', 'Gununghalu', 'Lembang', 'Ngamprah',
    ],
}

def _n(x):
    return str(x).lower().strip()

FLOODS = {yr: {_n(k) for k in ks} for yr, ks in FLOODS_RAW.items()}

FLOODS_COUNT_RAW = {
    2022: {
        'Mandalajati': 1, 'Bandung Wetan': 1, 'Bojongloa Kaler': 1,
        'Cimahi Utara': 6, 'Cimahi Tengah': 8, 'Cimahi Selatan': 7,
        'Arjasari': 1, 'Baleendah': 10, 'Bojongsoang': 18, 'Cangkuang': 1,
        'Cicalengka': 4, 'Cileunyi': 5, 'Ciparay': 6, 'Ciwidey': 4,
        'Dayeuhkolot': 18, 'Katapang': 2, 'Margaasih': 2, 'Nagreg': 4,
        'Pameungpeuk': 1, 'Pangalengan': 2, 'Pasirjambu': 2, 'Rancaekek': 12,
        'Solokanjeruk': 2, 'Soreang': 5, 'Ngamprah': 7, 'Padalarang': 1, 'Parongpong': 1,
    },
    2023: {
        'Bandung Kulon': 1, 'Sukasari': 1,
        'Cimahi Selatan': 10, 'Cimahi Tengah': 1, 'Cimahi Utara': 2,
        'Arjasari': 1, 'Baleendah': 6, 'Banjaran': 2, 'Bojongsoang': 11,
        'Cangkuang': 5, 'Cicalengka': 3, 'Cimenyan': 1, 'Ciparay': 5,
        'Ciwidey': 1, 'Dayeuhkolot': 17, 'Ibun': 2, 'Katapang': 3,
        'Kertasari': 2, 'Kutawaringin': 4, 'Majalaya': 3, 'Margaasih': 7,
        'Pameungpeuk': 3, 'Pangalengan': 2, 'Paseh': 4, 'Rancaekek': 3,
        'Soreang': 4, 'Cihampelas': 1, 'Cisarua': 1, 'Padalarang': 1, 'Sindangkerta': 2,
    },
    2024: {
        'Antapani': 2, 'Arcamanik': 2, 'Bandung Kidul': 1, 'Batununggal': 1,
        'Buahbatu': 1, 'Cibeunying Kaler': 2, 'Cibeunying Kidul': 1,
        'Gedebage': 2, 'Mandalajati': 1, 'Panyileukan': 2, 'Sumur Bandung': 1,
        'Cimahi Tengah': 1, 'Cimahi Utara': 4,
        'Arjasari': 3, 'Baleendah': 10, 'Banjaran': 7, 'Bojongsoang': 15,
        'Cangkuang': 3, 'Cicalengka': 8, 'Ciparay': 3, 'Ciwidey': 2,
        'Dayeuhkolot': 17, 'Ibun': 1, 'Katapang': 8, 'Kertasari': 5,
        'Majalaya': 2, 'Nagreg': 4, 'Pacet': 2, 'Pameungpeuk': 6,
        'Pangalengan': 3, 'Paseh': 3, 'Pasirjambu': 1, 'Rancaekek': 3,
        'Solokanjeruk': 3, 'Soreang': 2,
        'Cikalongwetan': 2, 'Cipatat': 2, 'Cipongkor': 1, 'Gununghalu': 1,
        'Lembang': 2, 'Ngamprah': 2,
    },
}
FLOODS_COUNT = {yr: {_n(k): v for k, v in d.items()} for yr, d in FLOODS_COUNT_RAW.items()}

def compute(gt, pred):
    tp = sum(1 for g, p in zip(gt, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt, pred) if g == 0 and p == 1)
    tn = sum(1 for g, p in zip(gt, pred) if g == 0 and p == 0)
    fn = sum(1 for g, p in zip(gt, pred) if g == 1 and p == 0)
    n  = tp + fp + tn + fn
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    # Cohen's Kappa
    po = (tp + tn) / n
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return tp, fp, tn, fn, po * 100, pr * 100, rc * 100, f1 * 100, kappa

def w_recall(names, gt, pred, year):
    """Recall berbobot: setiap kecamatan dibobotkan frekuensi kejadian banjirnya."""
    cnt = FLOODS_COUNT[year]
    w_tp, w_pos = 0, 0
    for kec, g, p in zip(names, gt, pred):
        if g == 1:
            w = cnt.get(_n(kec), 1)
            w_pos += w
            if p == 1:
                w_tp += w
    return w_tp / w_pos if w_pos > 0 else 0.0

perubahan_df = pd.read_csv('perubahan_cluster_antar_tahun.csv')
pca_df       = pd.read_csv('New PCA El n Sil/main_data_with_pred(0,31).csv')

UMAP_RAWAN    = {1, 2, 3, 4, 6, 7}
PCA_EL_RAWAN  = {1, 2}
PCA_SIL_RAWAN = {0}

SEP = '=' * 90

print(SEP)
print('  VALIDASI GROUND TRUTH — KLASTERISASI BANJIR BANDUNG RAYA')
print(SEP)
print(f"  {'Tahun':<8} {'Model':<18} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}"
      f"  {'Akurasi':>8} {'Presisi':>8} {'Recall':>8} {'F1-Score':>8} {'Kappa':>7} {'W-Recall':>9}")
print('-' * 90)

all_results = {}

for year in [2022, 2023, 2024]:
    yr_pca = pca_df[pca_df['year'] == year]

    # UMAP
    names_u = [r['kecamatan'] for _, r in perubahan_df.iterrows()]
    gt      = [1 if _n(n) in FLOODS[year] else 0 for n in names_u]
    pred    = [1 if int(r['cluster_2022']) in UMAP_RAWAN else 0 for _, r in perubahan_df.iterrows()]
    res_u   = compute(gt, pred)
    wr_u    = w_recall(names_u, gt, pred, year)
    all_results[('UMAP', year)] = res_u
    print(f"  {year:<8} {'UMAP':<18} {res_u[0]:>4} {res_u[1]:>4} {res_u[2]:>4} {res_u[3]:>4}"
          f"  {res_u[4]:>7.2f}% {res_u[5]:>7.2f}% {res_u[6]:>7.2f}% {res_u[7]:>7.2f}% {res_u[8]:>7.4f} {wr_u*100:>8.2f}%")

    # PCA Elbow
    kc      = yr_pca.groupby('kecamata')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
    names_e = [r['kecamata'] for _, r in kc.iterrows()]
    gt      = [1 if _n(n) in FLOODS[year] else 0 for n in names_e]
    pred    = [1 if int(r['cluster']) in PCA_EL_RAWAN else 0 for _, r in kc.iterrows()]
    res_e   = compute(gt, pred)
    wr_e    = w_recall(names_e, gt, pred, year)
    all_results[('PCA Elbow', year)] = res_e
    print(f"  {'':<8} {'PCA Elbow':<18} {res_e[0]:>4} {res_e[1]:>4} {res_e[2]:>4} {res_e[3]:>4}"
          f"  {res_e[4]:>7.2f}% {res_e[5]:>7.2f}% {res_e[6]:>7.2f}% {res_e[7]:>7.2f}% {res_e[8]:>7.4f} {wr_e*100:>8.2f}%")

    # PCA Silhouette
    kc      = yr_pca.groupby('kecamata')['cluster_sil'].agg(lambda x: x.mode()[0]).reset_index()
    names_s = [r['kecamata'] for _, r in kc.iterrows()]
    gt      = [1 if _n(n) in FLOODS[year] else 0 for n in names_s]
    pred    = [1 if int(r['cluster_sil']) in PCA_SIL_RAWAN else 0 for _, r in kc.iterrows()]
    res_s   = compute(gt, pred)
    wr_s    = w_recall(names_s, gt, pred, year)
    all_results[('PCA Silhouette', year)] = res_s
    print(f"  {'':<8} {'PCA Silhouette':<18} {res_s[0]:>4} {res_s[1]:>4} {res_s[2]:>4} {res_s[3]:>4}"
          f"  {res_s[4]:>7.2f}% {res_s[5]:>7.2f}% {res_s[6]:>7.2f}% {res_s[7]:>7.2f}% {res_s[8]:>7.4f} {wr_s*100:>8.2f}%")

    print()

# Gabungan
print('-' * 90)
print('  GABUNGAN 2022–2024')
print('-' * 90)

for model, rawan_set, col in [
    ('UMAP',          UMAP_RAWAN,    None),
    ('PCA Elbow',     PCA_EL_RAWAN,  'cluster'),
    ('PCA Silhouette',PCA_SIL_RAWAN, 'cluster_sil'),
]:
    gt_all, pred_all = [], []
    w_tp_all, w_pos_all = 0, 0
    for year in [2022, 2023, 2024]:
        cnt = FLOODS_COUNT[year]
        if model == 'UMAP':
            names_yr = [r['kecamatan'] for _, r in perubahan_df.iterrows()]
            gt_yr    = [1 if _n(n) in FLOODS[year] else 0 for n in names_yr]
            pred_yr  = [1 if int(r['cluster_2022']) in rawan_set else 0 for _, r in perubahan_df.iterrows()]
        else:
            yr_pca   = pca_df[pca_df['year'] == year]
            kc       = yr_pca.groupby('kecamata')[col].agg(lambda x: x.mode()[0]).reset_index()
            names_yr = [r['kecamata'] for _, r in kc.iterrows()]
            gt_yr    = [1 if _n(n) in FLOODS[year] else 0 for n in names_yr]
            pred_yr  = [1 if int(r[col]) in rawan_set else 0 for _, r in kc.iterrows()]
        gt_all   += gt_yr
        pred_all += pred_yr
        for kec, g, p in zip(names_yr, gt_yr, pred_yr):
            if g == 1:
                w = cnt.get(_n(kec), 1)
                w_pos_all += w
                if p == 1:
                    w_tp_all += w
    res    = compute(gt_all, pred_all)
    wr_gab = w_tp_all / w_pos_all if w_pos_all > 0 else 0.0
    print(f"  {'':<8} {model:<18} {res[0]:>4} {res[1]:>4} {res[2]:>4} {res[3]:>4}"
          f"  {res[4]:>7.2f}% {res[5]:>7.2f}% {res[6]:>7.2f}% {res[7]:>7.2f}% {res[8]:>7.4f} {wr_gab*100:>8.2f}%")

print(SEP)

# ── ARI Pendekatan B: label klaster asli vs ground truth biner ───────────────
print()
print(SEP)
print('  ARI (ADJUSTED RAND INDEX) — PENDEKATAN B')
print('  Label klaster asli (0-7 / 0-3 / 0-1) vs ground truth biner (banjir/tidak)')
print(SEP)
print(f"  {'Tahun':<8} {'Model':<18} {'Klaster':<10} {'ARI':>8}")
print('-' * 90)

for year in [2022, 2023, 2024]:
    yr_pca = pca_df[pca_df['year'] == year]

    # UMAP: label asli 0-7
    gt_u  = [1 if _n(r['kecamatan']) in FLOODS[year] else 0 for _, r in perubahan_df.iterrows()]
    lbl_u = [int(r['cluster_2022']) for _, r in perubahan_df.iterrows()]
    print(f"  {year:<8} {'UMAP':<18} {'0-7':<10} {adjusted_rand_score(gt_u, lbl_u):>8.4f}")

    # PCA Elbow: label asli 0-3
    kc    = yr_pca.groupby('kecamata')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
    gt_e  = [1 if _n(r['kecamata']) in FLOODS[year] else 0 for _, r in kc.iterrows()]
    lbl_e = [int(r['cluster']) for _, r in kc.iterrows()]
    print(f"  {'':<8} {'PCA Elbow':<18} {'0-3':<10} {adjusted_rand_score(gt_e, lbl_e):>8.4f}")

    # PCA Silhouette: label asli 0-1
    kc    = yr_pca.groupby('kecamata')['cluster_sil'].agg(lambda x: x.mode()[0]).reset_index()
    gt_s  = [1 if _n(r['kecamata']) in FLOODS[year] else 0 for _, r in kc.iterrows()]
    lbl_s = [int(r['cluster_sil']) for _, r in kc.iterrows()]
    print(f"  {'':<8} {'PCA Silhouette':<18} {'0-1':<10} {adjusted_rand_score(gt_s, lbl_s):>8.4f}")

    print()

# Gabungan
print('-' * 90)
print('  GABUNGAN 2022-2024')
print('-' * 90)

for model, col, klaster in [
    ('UMAP',           None,          '0-7'),
    ('PCA Elbow',      'cluster',     '0-3'),
    ('PCA Silhouette', 'cluster_sil', '0-1'),
]:
    gt_all, lbl_all = [], []
    for year in [2022, 2023, 2024]:
        if model == 'UMAP':
            gt_all  += [1 if _n(r['kecamatan']) in FLOODS[year] else 0 for _, r in perubahan_df.iterrows()]
            lbl_all += [int(r['cluster_2022']) for _, r in perubahan_df.iterrows()]
        else:
            yr_pca = pca_df[pca_df['year'] == year]
            kc = yr_pca.groupby('kecamata')[col].agg(lambda x: x.mode()[0]).reset_index()
            gt_all  += [1 if _n(r['kecamata']) in FLOODS[year] else 0 for _, r in kc.iterrows()]
            lbl_all += [int(r[col]) for _, r in kc.iterrows()]
    print(f"  {'':<8} {model:<18} {klaster:<10} {adjusted_rand_score(gt_all, lbl_all):>8.4f}")

print(SEP)
