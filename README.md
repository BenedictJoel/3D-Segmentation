# Panduan Demo — SAM-Med3D & nnU-Net untuk Segmentasi Jantung (CMRxMotion / Data Baru)

Panduan ini mencakup **dua notebook**:

| Notebook | Untuk apa |
|---|---|
| **`SAM_Med3D_Demo_Turbo_vs_FT_2.ipynb`** | Demo di **dataset CMRxMotion** (data berlabel). Membandingkan model **Zero-shot (Turbo)** vs **Fine-tuned**, plus figur-figur diagnostik untuk paper. |
| **`SAM_Med3D_nnUNet_RSHS_demo.ipynb`** | Mencoba model pada **data baru / RSHS** (`.nii`), **dengan atau tanpa label**. Termasuk **nnU-Net** otomatis dan **ekspor hasil ke `.nii`**. |

Keduanya memakai model yang **sudah dilatih** — tidak ada training ulang. Cukup dimuat lalu dipakai.

---

## 0. Prasyarat (berlaku untuk kedua notebook)

**Perangkat keras:** GPU NVIDIA (dipakai: RTX 4070 SUPER 12 GB). Tanpa GPU tetap jalan tapi lambat.

**Software:** Python 3.10+, dan paket:
`torch` (versi CUDA), `nibabel`, `numpy`, `pandas`, `scipy`, `matplotlib`, `ipywidgets`, `ipympl`.
Untuk notebook RSHS tambahkan `nnunetv2`.

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install nibabel numpy pandas scipy matplotlib ipywidgets ipympl nnunetv2
```

> **`ipympl` wajib.** Tanpa paket ini, `%matplotlib widget` (kanvas klik) tidak muncul. Jika UI tidak tampil: install `ipympl` lalu **restart kernel**.

**Struktur folder** (ganti `D:\Joel` ke lokasimu):

```
D:\Joel\
├─ Dataset\data\                     # citra + label .nii.gz (CMRxMotion)
├─ SAM-Med3D\                         # repo SAM-Med3D (hasil clone)
│   └─ ckpt\sam_med3d_turbo.pth       # checkpoint Turbo (zero-shot)
├─ work\
│   ├─ sam_ft_frozen\fold0..4\sam_model_best.pth   # checkpoint fine-tuned per fold
│   ├─ results_noc\                   # folds_subjek.csv + kurva CSV + output demo CMRxMotion
│   ├─ results_rshs\                  # output demo data baru (otomatis dibuat)
│   └─ nnunet\nnUNet_results\         # model nnU-Net terlatih
└─ RSHS\                              # tempat menaruh data baru (.nii)
```

> **PENTING — file besar tidak ada di GitHub.** Checkpoint Turbo dan model nnU-Net (± 5 GB) **tidak** ikut di repo. Notebook hanya jalan di laptop yang berisi file-file ini. Kalau dijalankan di komputer lain, **kedua file itu harus disalin** ke sana lebih dulu.

---

## A. Notebook Demo CMRxMotion — `SAM_Med3D_Demo_Turbo_vs_FT_2.ipynb`

Tujuan: menunjukkan **cara setup** dan membandingkan Zero-shot vs Fine-tuned secara interaktif. **Demo inti = jalankan Sel 1 → 2 → 3 → 4B** secara berurutan.

### Langkah demo (wajib, berurutan)

1. **Sel "1. Konfigurasi + data"** — import paket dan menetapkan semua path + konstanta (`SAM_SPACING=1.5mm`, `SAM_ROI=128`, checkpoint, `folds_subjek.csv`).
   - **Ganti `D:\Joel\...`** ke foldermu sebelum run.
2. **Sel "2. Preprocessing, metrik, oracle"** — fungsi resample ke 1.5 mm isotropik, crop ROI 128³ di centroid GT, z-score, Dice, dan oracle klik.
3. **Sel "3. Model manager"** — memuat model `vit_b_ori`; `set_model('turbo'|'finetuned', fold)`. Angka "hilang/lebih" saat load itu **normal** (kita hanya pakai sebagian bobot).
4. **Sel "4b. Demo dual manual"** — UI utama. Dua panel: kiri Zero-shot (Turbo), kanan Fine-tuned.

### Cara pakai UI 4B

1. Pilih **Kasus** dan **Struktur** (LV / MYO / RV) → klik **Muat**.
2. Geser **Slice** ke irisan jantung yang pas.
3. Pilih **Positif +** (menandai di dalam target) atau **Negatif −** (membuang area salah), lalu **klik langsung di gambar**. Satu klik diterapkan ke **kedua model** sekaligus.
4. **Klik tengah** = satu klik positif di pusat; **Undo** = batal klik terakhir; **Reset** = kosongkan; centang **GT** = tampilkan kontur hijau ground-truth.
5. Judul tiap panel menampilkan **DSC** real-time. Puas → **Render & Simpan PNG** (disimpan di `work\results_noc\`).

### Sel opsional (tidak perlu saat demo langsung)

Sel 5–9 untuk **menghasilkan figur paper**, bukan demo interaktif:
- **5 / 5b** — diagnostik oracle Turbo vs FT (grid 3×2 per struktur).
- **6** — angka agregat pada subjek test satu fold (set `RUN_AGG=True`, agak lama).
- **7** — kurva DSC-vs-klik + garis nnU-Net + bar reach/NoC@0.85 (dibaca dari CSV di `results_noc\`, **tanpa** inferensi ulang).
- **8 / 9** — pasangan citra asli vs segmentasi / vs GT native.

> Catatan: `folds_subjek.csv` dipakai agar tiap kasus memakai **checkpoint fold yang benar** — model yang tidak melihat subjek itu saat training, supaya evaluasi jujur (tanpa kebocoran data).

---

## B. Notebook Data Baru / RSHS — `SAM_Med3D_nnUNet_RSHS_demo.ipynb`

Tujuan: mencoba model pada **data baru `.nii`**. Dua jalur, dipilih lewat satu baris konfigurasi:

- **MODE A — ADA GT** (data berlabel, untuk validasi): ROI dari centroid GT, DSC & kontur hijau tampil.
- **MODE B — TANPA GT** (data klinis mentah): pusat ROI ditentukan **manual lewat klik**, prediksi saja.

### Menyiapkan data

Taruh file di `D:\Joel\RSHS\`, misalnya `case01.nii.gz`. Kalau ada label, tambahkan `case01-label.nii.gz` (kode kelas **1=LV, 2=MYO, 3=RV**, satu geometri dengan citra).

> Data **DICOM** harus dikonversi ke `.nii` dulu (mis. `dcm2niix`), karena notebook membaca NIfTI, bukan `.dcm`. Model ini **3D**, jadi butuh **satu volume/series** — bukan satu irisan 2D saja. Citra 4D (cine) otomatis diambil frame pertama.

### Konfigurasi (Sel "Langkah 1")

```python
RSHS_IMAGE  = Path(r"D:\Joel\RSHS\case01.nii.gz")   # citra, wajib
RSHS_LABEL  = None   # None = MODE B (tanpa GT). Isi Path(...) utk MODE A (ada GT).
RSHS_STRUCT = 1      # 1=LV, 2=MYO, 3=RV
FT_FOLD     = 0      # checkpoint fine-tuned fold mana (0..4)
```

Ada pengecekan di awal: kalau citra/label/checkpoint tidak ada, atau ukuran citra ≠ label, langsung muncul pesan yang jelas.

### Menjalankan

1. Jalankan sel dari atas (Langkah 1 → 2 → 3).
2. **MODE B (tanpa GT):** di **Langkah 4** muncul picker — geser slice ke jantung, **klik satu kali di pusat jantung**, tekan **Pakai pusat terpilih**.
3. **Langkah 5 — demo klik SAM.** Di MODE B, jika baru memilih pusat setelah sel ini jalan, tekan **Muat** agar ROI dipotong ulang di pusat baru. Klik **+** untuk membentuk masker, **−** untuk mengoreksi.
4. **Ekspor .nii** — menyimpan prediksi Turbo & Fine-tuned ke **geometri asli pasien** di `work\results_rshs\`. Buka di **3D Slicer / ITK-SNAP** agar menempel pas pada citra asli.

### nnU-Net otomatis (Langkah 6)

Dua cara: jalankan `nnUNetv2_predict` (CLI), atau muat hasil `.nii` yang sudah ada (`NNUNET_OUT`). Konfigurasi harus cocok dengan model terlatih di `work\nnunet`:

```python
NNUNET_RESULTS = WORK_DIR/"nnunet"/"nnUNet_results"
NNUNET_DATASET = "Dataset111_CMRxMotion"
NNUNET_TRAINER = "nnUNetTrainer_100epochs"
NNUNET_CONFIG  = "3d_fullres"
NNUNET_FOLDS   = "0"          # atau "0 1 2 3 4" untuk ensemble (lebih lama)
```

Setelah hasil ada, sel viewer menampilkan overlay nnU-Net per kelas (LV merah, MYO kuning, RV cyan). Pesan error lengkap nnU-Net tampil di output sel bila gagal.

### Status uji (sudah diverifikasi)

- **MODE A** (citra 3D + label): semua tahap jalan — Turbo vs FT, DSC (contoh: ZS 0,854 / FT 0,917 dengan 1 klik), ekspor `.nii`, PNG, nnU-Net. Ekspor menempel pas di geometri asli (DSC di ruang asli 0,925).
- **MODE B** (citra 4D tanpa label): picker pusat, prediksi, ekspor, dan nnU-Net semuanya jalan.
- Hasil data baru disimpan terpisah di `work\results_rshs\`, tidak tercampur dengan hasil paper di `results_noc\`.

---

## Catatan penting

- **Hanya jalan di laptop yang berisi checkpoint Turbo + model nnU-Net (± 5 GB).** File ini tidak ada di GitHub; salin dulu bila pindah komputer.
- **Domain shift:** kedua model dilatih di CMRxMotion. Pada scanner berbeda (RSHS) hasil bisa menurun. **SAM-Med3D bisa dikoreksi dengan klik; nnU-Net otomatis tidak.** Jangan janjikan akurasi setara.
- **Pemilihan fold FT** pada data baru: fold mana pun valid (default `FT_FOLD=0`); bandingkan beberapa fold bila ingin lebih kokoh.
- **Pusat ROI (MODE B):** kubus 128³ ≈ 19 cm pada 1.5 mm. Kalau jantung tidak masuk penuh, geser pusat / pilih slice lain saat memilih.

## Troubleshooting ringkas

| Gejala | Penyebab / solusi |
|---|---|
| UI/kanvas tidak muncul | `ipympl` belum ada → install, **restart kernel**, pastikan `%matplotlib widget` jalan |
| `checkpoint tak ada: ...` | path salah atau file `.pth` belum disalin ke tempatnya |
| `ModuleNotFoundError: segment_anything` | `SAM_REPO` tidak menunjuk folder repo SAM-Med3D |
| `CUDA out of memory` | tutup proses lain yang pakai GPU, atau jalankan di CPU (lebih lambat) |
| nnU-Net `command not found` / gagal | `pip install nnunetv2`; `-d/-tr/-c/-f` harus cocok dengan model terlatih; cek `nnUNet_results` |
| "GT kosong untuk struktur ini" | struktur itu tidak ada pada frame terpilih; ganti kasus/struktur |
