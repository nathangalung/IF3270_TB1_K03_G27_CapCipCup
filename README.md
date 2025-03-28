# Tugas Besar 1 IF3270 - Feed Forward Neural Network

## Kelompok 27 - CapCipCup

- Jonathan Emmanuel Saragih (13522121)
- Jason Fernando (13522156)
- Bryan P. Hutagalung (18222130)

Program Studi Teknik Informatika  
Sekolah Teknik Elektro dan Informatika  
Institut Teknologi Bandung

---

## Judul Proyek

**Implementasi dan Eksperimen Feed Forward Neural Network dari Awal**

---

## Deskripsi

Tugas besar ini merupakan implementasi dari model **Feed Forward Neural Network (FFNN)** secara manual tanpa menggunakan framework deep learning seperti TensorFlow atau PyTorch (kecuali untuk perbandingan dan pembacaan dataset). Proyek ini bertujuan untuk memahami konsep dasar jaringan saraf tiruan dan bagaimana komponen-komponen seperti fungsi aktivasi, loss function, forward-backward propagation, hingga hyperparameter tuning diimplementasikan dari awal.

---

## Spesifikasi Utama

Model FFNN yang dikembangkan memiliki fitur sebagai berikut:

- **Struktur Jaringan**:

  - Jumlah neuron per layer dapat diatur bebas.
  - Dukungan berbagai fungsi aktivasi: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Softplus`, `ELU`.

- **Fungsi Loss**:

  - Mean Squared Error (MSE)
  - Binary Cross-Entropy
  - Categorical Cross-Entropy

- **Inisialisasi Bobot**:

  - Zero Initialization
  - Random Uniform/Normal (dengan parameter bound, mean, std, dan seed)
  - Xavier Initialization
  - He Initialization

- **Training dan Evaluasi**:

  - Dukungan batch training
  - Parameter: learning rate, batch size, epoch, verbose
  - Logging training dan validation loss
  - Evaluasi dengan metrik: accuracy, precision, recall, F1-score, confusion matrix

- **Regularisasi dan Normalisasi**:

  - L1 & L2 Regularization
  - RMS Normalization

- **Fitur Tambahan**:
  - Visualisasi loss dan distribusi bobot/gradien
  - Simpan/muat model (`save` / `load`)
  - Gradient clipping
  - Early stopping
  - Learning rate scheduler

---

## Dataset

Dataset yang digunakan untuk pelatihan dan pengujian model adalah **MNIST (mnist_784)**, diambil menggunakan fungsi `fetch_openml` dari `sklearn.datasets`.

---

## Struktur Proyek

````bash
.
├── main.py                  # Script utama
├── models/                 # Implementasi model dan layer
│   ├── activations.py
│   ├── layers.py
│   ├── network.py
│   └── initializers.py
├── trainers/               # Trainer dan utilitas training
│   ├── trainer.py
│   ├── early_stopping.py
│   └── scheduler.py
├── utils/                  # Fungsi bantu: visualisasi, evaluasi, dll.
│   ├── visualization.py
│   └── metrics.py
├── notebooks/              # Notebook pengujian dan eksperimen
│   └── testing_ffnn.ipynb
├── README.md
└── requirements.txt

---

## Cara Menjalankan

1. **Clone repository atau salin file proyek** ke dalam direktori lokal.

2. **(Opsional) Install dependencies:**
   Jika menggunakan virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # Linux/macOS
   env\Scripts\activate.bat  # Windows
   pip install -r requirements.txt

3. Jalankan script utama:
   ```bash
   python main.py
````

atau buka notebook `notebooks/testing_ffnn.ipynb` dan melakukan proses run

## Eksperimen dan Pengujuan

# Model diuji menggunakan dataset MNIST dan dilakukan analisis terhadap beberapa aspek berikut:

a. Depth & Width: Menguji efek jumlah layer dan jumlah neuron pada performa model.
b. Fungsi Aktivasi: ReLU, Sigmoid, Tanh, Linear, Softplus, ELU dibandingkan berdasarkan akurasi dan loss.
c. Learning Rate: Eksperimen 3 nilai berbeda dan pengaruhnya terhadap konvergensi.
d. Metode Inisialisasi: Dibandingkan Zero, Random Uniform/Normal, Xavier, dan He.
e. Regularisasi: Model diuji dengan L1, L2, dan tanpa regularisasi.
f. RMS Normalization: Pengaruh normalisasi terhadap stabilitas training.
g. Perbandingan dengan sklearn.MLPClassifier: Untuk validasi kebenaran implementasi.

# Hasil evaluasi mencakup:

a. Grafik training dan validation loss
b. Visualisasi distribusi bobot dan gradien
c. Confusion matrix dan metrik evaluasi: Accuracy, Precision, Recall, F1-Score
