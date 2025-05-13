# Predictiveanalytics

# Laporan Proyek Machine Learning - Wahyu Ozorah Manurung

## Domain Proyek: Pendidikan

### Latar Belakang
Tingkat kelulusan mahasiswa merupakan indikator utama dalam menilai kualitas dan efektivitas institusi pendidikan tinggi. Di Indonesia, pencapaian akreditasi institusi sangat dipengaruhi oleh rasio kelulusan tepat waktu, yang mencerminkan efisiensi proses pembelajaran dan keberhasilan akademik mahasiswa. Namun, banyak perguruan tinggi masih menghadapi tantangan dalam mengidentifikasi mahasiswa yang berisiko tidak lulus tepat waktu, terutama karena keterbatasan dalam sistem pemantauan dan intervensi dini.

Seiring dengan meningkatnya ketersediaan data akademik dan kemajuan dalam bidang data science, pendekatan berbasis machine learning telah menjadi solusi potensial untuk memprediksi kelulusan mahasiswa. Teknik ini memungkinkan analisis terhadap berbagai faktor, seperti prestasi akademik, kehadiran, latar belakang sosial-ekonomi, dan aktivitas ekstrakurikuler, guna mengidentifikasi pola yang memengaruhi kelulusan.

Beberapa penelitian telah menunjukkan efektivitas penggunaan algoritma machine learning dalam memprediksi kelulusan mahasiswa. Misalnya, studi oleh Saki (2023) mengembangkan model prediktif menggunakan algoritma decision tree dan random forest untuk mengklasifikasikan status kelulusan mahasiswa berdasarkan data akademik dan non-akademik. Hasilnya menunjukkan bahwa model tersebut dapat membantu institusi pendidikan dalam mengidentifikasi mahasiswa yang memerlukan intervensi dini.

Selain itu, penelitian oleh Kim et al. (2023) menggunakan berbagai teknik machine learning untuk menganalisis faktor-faktor yang berkontribusi terhadap risiko mahasiswa tidak lulus, seperti data akademik, demografi, dan sosial-ekonomi. Studi ini menekankan pentingnya data akademik dalam meningkatkan akurasi prediksi kelulusan.

Dengan memanfaatkan pendekatan ini, institusi pendidikan dapat mengembangkan sistem prediktif yang membantu dalam pengambilan keputusan strategis, seperti penyusunan program bimbingan akademik, penyesuaian kurikulum, dan alokasi sumber daya. Hal ini tidak hanya meningkatkan tingkat kelulusan, tetapi juga memperkuat reputasi institusi dan memenuhi standar akreditasi nasional.

### Referensi
A. Saki, Student Graduation Result Prediction, 2023. , S. Kim, E. Yoo, & S. Kim, Why Do Students Drop Out?, arXiv:2310.10987, 2023. [https://arxiv.org/abs/2310.10987], Polinela Research Team. (2023). Student Graduation Prediction Using Machine Learning Algorithms. Jurnal Routers, 10(2), 45-56. [https://jurnal.polinela.ac.id/routers/article/view/3897]


## Business Understanding

Untuk meningkatkan efisiensi akademik dan mendukung mahasiswa agar dapat lulus tepat waktu, diperlukan sistem yang mampu memprediksi kelulusan mahasiswa secara akurat berdasarkan data historis. Analisis ini akan membantu institusi pendidikan melakukan intervensi lebih awal dan mengambil keputusan yang tepat.

### Problem Statements

- Bagaimana memprediksi status kelulusan mahasiswa (lulus/tidak lulus) berdasarkan data akademik dan non-akademik seperti GPA, absensi, waktu belajar, dan faktor lainnya?
- Fitur mana yang paling berpengaruh terhadap kemungkinan mahasiswa lulus atau tidak lulus?
- Model klasifikasi machine learning mana yang memberikan hasil prediksi terbaik dalam kasus ini?

### Goals

- Membangun model klasifikasi yang dapat memprediksi status kelulusan mahasiswa dengan tingkat akurasi yang baik berdasarkan fitur-fitur yang tersedia dalam dataset.
- Mengidentifikasi variabel-variabel utama yang berpengaruh signifikan terhadap status kelulusan mahasiswa, sehingga dapat dijadikan dasar untuk kebijakan akademik.
- Membandingkan performa beberapa algoritma machine learning (Decision Tree C4.5, Random Forest, dan SVM) dan memilih model dengan evaluasi terbaik berdasarkan metrik seperti akurasi, precision, recall, dan f1-score.
- Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.


### Solution statements
- Menggunakan tiga algoritma machine learning: Decision Tree C4.5, Random Forest, dan Support Vector Machine (SVM) untuk membangun model prediksi kelulusan. Setiap model akan dilatih dan dievaluasi menggunakan data yang telah dibersihkan dan dinormalisasi.

- Melakukan tuning hyperparameter menggunakan GridSearchCV untuk setiap algoritma agar memperoleh performa model yang optimal dan mengurangi risiko overfitting/underfitting.

- Menggunakan metrik evaluasi yang terukur, seperti:
  A. Accuracy: untuk mengukur keseluruhan kinerja model.
  B.Precision dan Recall: untuk mengevaluasi model terhadap kelas yang tidak seimbang.
  C.F1-score: untuk menyeimbangkan precision dan recall dalam satu metrik.
  D.Confusion matrix: untuk melihat distribusi kesalahan klasifikasi model.
- Melakukan analisis feature importance (khususnya pada model Random Forest) untuk menentukan fitur mana yang paling berpengaruh terhadap status kelulusan.

## Data Understanding
Proyek ini menggunakan dataset yang diperoleh dari Kaggle dengan judul "Student Performance Dataset" yang dapat diakses melalui tautan berikut:
🔗 https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset 

Dataset ini menyajikan informasi lengkap tentang karakteristik mahasiswa yang mencakup aspek akademik, sosial, serta dukungan dari lingkungan sekitarnya. Data ini digunakan untuk membangun model prediktif yang menentukan status kelulusan mahasiswa berdasarkan fitur-fitur yang tersedia.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- StudentID: ID unik untuk setiap mahasiswa.
- Age: Usia mahasiswa.
- Gender: Jenis kelamin mahasiswa (Laki-laki/Perempuan).
- Ethnicity: Kelompok etnis mahasiswa.
- ParentalEducation: Tingkat pendidikan tertinggi orang tua.
- StudyTimeWeekly: Jumlah waktu belajar rata-rata per minggu (dalam jam).
- Absences: Jumlah ketidakhadiran mahasiswa dalam kelas.
- Tutoring: Indikator apakah mahasiswa mengikuti program bimbingan belajar tambahan (Yes/No).
- ParentalSupport: Menunjukkan apakah orang tua secara aktif mendukung kegiatan belajar (Yes/No).
- Extracurricular: Keterlibatan dalam kegiatan ekstrakurikuler.
- Sports: Keterlibatan dalam kegiatan olahraga.
- Music: Partisipasi dalam kegiatan musik.
- Volunteering: Keterlibatan dalam kegiatan sukarelawan.
- GPA: Nilai rata-rata akademik mahasiswa (Grade Point Average).
- GradeClass: Kategori nilai akhir mahasiswa dalam bentuk kelas (telah dikonversi ke angka dalam analisis).

Beberapa tahapan eksplorasi data telah dilakukan untuk memahami karakteristik dataset, antara lain:
- Dataset dicek secara menyeluruh dan ditemukan bahwa tidak ada nilai yang hilang maupun data yang duplikat. Ini menunjukkan bahwa data dalam kondisi baik untuk digunakan dalam pemodelan tanpa perlu imputasi atau pembersihan lanjutan.

![image](https://github.com/user-attachments/assets/9e22b950-2a3f-4f5f-a8ff-104fd8bfc639)
  
dan menghilangkan data duplikat

![image](https://github.com/user-attachments/assets/54b8daf9-05b5-4db7-949d-5f645cc70ab9)

- Pendeteksian Outlier: Menggunakan metode IQR pada fitur numerik (StudyTimeWeekly, Absences, dan GPA) untuk mendeteksi nilai ekstrem.
![image](https://github.com/user-attachments/assets/7c34242e-527b-43e4-b694-20691d1db8ae)
Variabel numerik seperti GPA, StudyTimeWeekly, dan Absences diperiksa menggunakan metode IQR untuk mengidentifikasi nilai-nilai ekstrem. Outlier ini divisualisasikan melalui boxplot untuk membantu pengambilan keputusan apakah akan dihapus atau dipertahankan dalam pemodelan.

- Visualisasi Distribusi Kelas: Histogram dan boxplot menunjukkan hubungan antara waktu belajar, absensi, dan GPA dengan nilai akhir.
  ![image](https://github.com/user-attachments/assets/7e74cf50-1e9c-42ca-bf3a-82bac18a4283) , ![image](https://github.com/user-attachments/assets/067fa384-a5ee-472c-b3fa-29b6740d0792)

  Visualisasi berupa histogram dan boxplot digunakan untuk menampilkan distribusi GradeClass dan GraduationStatus. Hasil menunjukkan bahwa sebagian besar mahasiswa lulus, namun masih terdapat proporsi signifikan yang tidak lulus.
  
- Heatmap Korelasi: Mengungkap bahwa GPA dan Absences memiliki korelasi tertinggi terhadap status kelulusan.
  ![image](https://github.com/user-attachments/assets/9a1dcb0b-a691-4b0c-830e-b124df147698)
  Analisis korelasi menunjukkan bahwa GPA dan Absences memiliki hubungan paling kuat terhadap variabel target GraduationStatus. Ini menandakan bahwa performa akademik dan tingkat kehadiran adalah faktor utama yang memengaruhi kelulusan.

## Data Preparation
Data preparation sangat penting dalam pipeline machine learning karena memastikan data dalam kondisi bersih, terstruktur, dan siap digunakan oleh algoritma untuk pelatihan dan prediksi. Proses ini dilakukan secara bertahap dan sistematis agar menghasilkan model yang akurat dan dapat diinterpretasikan dengan baik.

1. Memeriksa Struktur Data dan Informasi Umum
   Langkah pertama dilakukan dengan fungsi df.info() dan df.isnull().sum() untuk memahami tipe data dan memastikan tidak ada nilai yang hilang (missing value). Selain itu, dilakukan pengecekan duplikasi     menggunakan df.duplicated().
  Alasan:
  Menjamin integritas data sangat penting sebelum analisis dilakukan. Kehadiran nilai hilang atau duplikat dapat menyebabkan bias atau error pada proses modeling.

2. Pendeteksian dan Visualisasi Outlier
   Outlier pada variabel numerik (StudyTimeWeekly, Absences, dan GPA) diperiksa menggunakan metode Interquartile Range (IQR). Nilai batas bawah dan atas dihitung, dan data di luar rentang ini dianggap sebagai   outlier. Visualisasi boxplot digunakan untuk melihat distribusi dan keberadaan outlier secara visual.
   Alasan:
   Outlier dapat memengaruhi performa model, terutama pada algoritma berbasis jarak atau statistik. Mendeteksinya memberi opsi untuk menghapus, mengubah, atau mempertahankannya berdasarkan konteks data.

3. Encoding Variabel Kategorikal
   Beberapa fitur memiliki nilai kategorikal seperti "Gender", "Ethnicity", "ParentalEducation", "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", dan "Volunteering". Variabel-variabel ini diubah menjadi format numerik menggunakan Label Encoding dari sklearn.preprocessing.LabelEncoder.
 ```python
    encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
 ```
  Alasan:
   Sebagian besar algoritma machine learning tidak dapat memproses nilai string secara langsung. Oleh karena itu, encoding diperlukan untuk mengubah data kategorikal menjadi representasi numerik yang bisa diproses model.
   
4. Normalisasi Fitur Numerik
   Fitur numerik seperti StudyTimeWeekly, Absences, dan GPA dinormalisasi menggunakan MinMaxScaler dari sklearn. Skala setiap nilai menjadi rentang antara 0 dan 1.
   
```python
   scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
 ```
Alasan:
Normalisasi membantu mempercepat proses pelatihan dan meningkatkan performa model, terutama untuk algoritma yang sensitif terhadap skala seperti SVM dan KNN.

5. Seleksi Fitur dan Pembuatan Dataset Terpilih
   Setelah melakukan eksplorasi korelasi, dipilih beberapa fitur penting yang memiliki hubungan kuat dengan target (GraduationStatus), yaitu:
   Data kemudian disimpan dalam file selected_features.csv untuk memudahkan proses selanjutnya.
```python
   selected_columns = ["StudyTimeWeekly", "Volunteering", "Tutoring", "ParentalEducation", "Absences", "GPA", "GradeClass"]
 ```
  Alasan:
  Pemilihan fitur penting (feature selection) membantu meningkatkan performa model, mengurangi overfitting, dan mempercepat waktu pelatihan dengan menghilangkan fitur yang kurang relevan.

6. Pembuatan Label Target (GraduationStatus)
   Label biner GraduationStatus dibuat dari kolom GradeClass. Jika nilai kelas termasuk A, B, atau C (dalam bentuk numerik: 0, 1, 2), maka mahasiswa dianggap lulus (1), selain itu dianggap tidak lulus (0).
```python
  df["GraduationStatus"] = df["GradeClass"].apply(lambda x: 1 if x in [0, 1, 2] else 0)
 ```
  Alasan:
  Label ini digunakan sebagai target variabel dalam klasifikasi biner untuk memprediksi apakah seorang mahasiswa akan lulus atau tidak.

7. Pembagian Data: Training dan Testing
   Data kemudian dipisahkan menjadi data latih dan data uji menggunakan fungsi train_test_split() dengan rasio 80:20 dan stratifikasi terhadap target.
```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
 ```
8. Penyesuaian Tipe Data
   Fitur kategorikal yang telah di-encode seperti "Volunteering", "Tutoring", dan "ParentalEducation" dikonversi ke tipe float64 untuk memastikan konsistensi dengan fitur numerik lainnya, terutama agar   kompatibel dengan algoritma tertentu.
   ```python
   X[col] = LabelEncoder().fit_transform(X[col]).astype(float)
    ```
  Alasan:
  Beberapa model memerlukan format numerik dengan tipe data float untuk perhitungan jarak atau matriks kernel, seperti pada SVM.
  
## Modeling
Tahapan modeling bertujuan untuk membangun model prediktif yang dapat menentukan status kelulusan mahasiswa (lulus atau tidak) berdasarkan fitur-fitur yang telah diproses sebelumnya. Dalam proyek ini digunakan tiga algoritma machine learning untuk dibandingkan:
- Decision Tree C4.5,
- Random Forest, dan
- Support Vector Machine (SVM).
Setiap algoritma diuji menggunakan stratified train-test split (80:20) dan dilakukan tuning hyperparameter menggunakan GridSearchCV untuk menemukan parameter optimal yang menghasilkan akurasi terbaik.

1. Decision Tree C4.5 – Parameter dan Tuning
   Model Decision Tree C4.5 dibangun menggunakan algoritma DecisionTreeClassifier dari pustaka Scikit-Learn. Untuk menghasilkan model yang optimal dan tidak overfitting, dilakukan proses hyperparameter tuning menggunakan GridSearchCV dengan beberapa kombinasi parameter penting.
```python
   param_grid = {
    'criterion': ['entropy'],
    'splitter': ['best'],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': ['balanced'],
    'ccp_alpha': [0.0, 0.01, 0.05]
}
```
> criterion='entropy' digunakan untuk mengukur kualitas pemisahan menggunakan informasi gain, sesuai dengan karakteristik C4.5 yang berbasis pada entropi.
> splitter='best' mengarahkan algoritma untuk memilih pemisahan terbaik pada setiap node.
> max_depth disesuaikan dalam beberapa pilihan seperti 3, 5, 10, 15, hingga None (tanpa batas), untuk menguji kedalaman pohon yang paling optimal. Semakin dalam pohon, semakin kompleks modelnya.
> min_samples_split dan min_samples_leaf mengontrol jumlah minimum sampel yang diperlukan untuk memisahkan node dan membentuk daun. Nilai seperti 2, 5, 10, dan 20 diuji untuk mencegah pembelahan yang terlalu agresif dan mengurangi overfitting.
> max_features mengatur jumlah fitur yang dipertimbangkan saat mencari pemisahan terbaik. Pengaturan seperti None, 'sqrt', dan 'log2' diuji untuk meningkatkan keragaman dalam pemisahan fitur.
> class_weight='balanced' digunakan untuk menyesuaikan bobot kelas jika terjadi ketidakseimbangan antara kelas lulus dan tidak lulus.
> Terakhir, ccp_alpha digunakan sebagai parameter pruning post-training (cost-complexity pruning), dengan nilai 0.0, 0.01, dan 0.05 untuk mengurangi kompleksitas pohon dan mencegah overfitting.
Proses tuning dilakukan dengan 5-fold cross-validation menggunakan metrik accuracy sebagai skor utama, untuk memilih kombinasi parameter terbaik yang memberikan generalisasi terbaik di data uji.

2. Random Forest
  Model Random Forest adalah metode ensemble learning yang membangun banyak pohon keputusan dan menggabungkan hasilnya. Untuk model ini digunakan RandomForestClassifier, dan dilakukan hyperparameter tuning   untuk menyesuaikan performa dan stabilitas.
```python
param_grid = {
    'n_estimators': [50, 100],
    'criterion': ['gini'],
    'max_depth': [5, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'class_weight': ['balanced']
}
```
> n_estimators adalah jumlah pohon dalam hutan. Nilai yang diuji adalah 50 dan 100. Semakin banyak pohon, biasanya semakin akurat model, namun juga lebih lambat secara komputasi.
> criterion='gini' digunakan untuk mengukur impurity di setiap node. Gini Impurity adalah metode yang umum digunakan karena komputasinya lebih cepat dari entropy.
> max_depth dibatasi pada 5 dan 10 untuk mencegah pembentukan pohon yang terlalu dalam dan kompleks, yang dapat menyebabkan overfitting.
> min_samples_split=2 adalah nilai default minimum jumlah sampel untuk memisahkan node. Ini membantu menjaga kedalaman pohon agar tidak terlalu tinggi.
> min_samples_leaf diuji dengan nilai 1 dan 2, untuk memastikan bahwa setiap daun memiliki cukup data yang mewakili distribusi target.
> max_features='sqrt' digunakan agar pada setiap pembentukan node, hanya subset akar kuadrat dari jumlah total fitur yang dipertimbangkan. Ini meningkatkan keragaman antar pohon dan mencegah korelasi antar fitur.
> class_weight='balanced' digunakan agar model memperhatikan ketidakseimbangan antara kelas lulus dan tidak lulus.

Tuning dilakukan menggunakan GridSearchCV dengan 5-fold cross-validation, di mana hasil menunjukkan bahwa Random Forest lebih stabil dan memberikan akurasi tertinggi dibanding model lain.

3. Support Vector Machine (SVM)
   Model SVM (Support Vector Machine) dibangun dengan menggunakan kelas SVC() dari Scikit-Learn. Karena SVM sangat sensitif terhadap skala data dan parameter, dilakukan hyperparameter tuning untuk menyesuaikan dengan karakteristik data.
```python
   param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
```

> C adalah parameter regularisasi yang mengontrol keseimbangan antara klasifikasi sempurna data pelatihan dan margin maksimum. Nilai yang diuji adalah 0.1, 1, 10, dan 100. Nilai C kecil membuat margin lebih lebar (lebih toleran terhadap kesalahan), sedangkan nilai besar mencoba mengklasifikasikan semua data pelatihan dengan benar.
> kernel menentukan jenis fungsi kernel yang digunakan untuk memproyeksikan data ke dimensi yang lebih tinggi. Kernel yang diuji meliputi 'linear', 'rbf', 'poly', dan 'sigmoid'.
> gamma adalah koefisien kernel untuk kernel 'rbf', 'poly', dan 'sigmoid'. Nilai 'scale' dan 'auto' diuji. Gamma menentukan jangkauan pengaruh satu titik data—gamma tinggi menyebabkan model lebih fokus pada titik-titik dekat (berisiko overfitting), sedangkan gamma rendah membuat model lebih umum.

Karena SVM membutuhkan data yang ternormalisasi, seluruh fitur numerik telah diskalakan menggunakan MinMaxScaler sebelumnya. Setelah tuning, SVM memberikan performa baik, namun dengan sedikit peningkatan kesalahan klasifikasi negatif.

**Kelebihan dan Kekurangan Setiap Algoritma**


Berikut adalah tabel Kelebihan dan Kekurangan dari ketiga algoritma:

| **Model**               | **Kelebihan**                                                                                                                  | **Kekurangan**                                       | 
|-------------------------|------------------------------------------------------------------------------------------------------------------------------- | -----------------------------------------------------|
| **Decision Tree C4.5**  |- Mudah diinterpretasikan dan divisualisasikan. <br> - Dapat menangani data numerik dan kategorikal. <br> - Cepat dilatih dan dieksekusi. | - Cenderung overfitting, terutama pada dataset yang kompleks. <br> - Performa menurun jika tidak dilakukan pruning atau tuning.
| **Random Forest**       |- Lebih stabil dan akurat daripada decision tree tunggal. <br> - Tahan terhadap overfitting karena menggunakan banyak pohon. <br> -Mampu menangani data yang tidak seimbang. | - Waktu pelatihan dan prediksi lebih lama dibanding decision tree. <br> Kurang mudah untuk diinterpretasikan (sebagai model ensemble).
| **Support Vector Machine (SVM)** | - Efektif untuk dataset berdimensi tinggi. <br> - Dapat bekerja dengan baik pada margin sempit antar kelas. <br> - Dapat menangani non-linearitas dengan kernel trick |- Waktu pelatihan lebih lama, terutama pada dataset besar.<br> - Sulit diinterpretasikan. <br> - performa menurun jika fitur tidak disesuaikan (scaling wajib).


## Evaluation
Setelah membangun model klasifikasi untuk memprediksi status kelulusan mahasiswa, langkah penting selanjutnya adalah mengevaluasi performa model dengan menggunakan metrik yang sesuai. Karena permasalahan ini merupakan klasifikasi biner (lulus atau tidak lulus), maka metrik evaluasi yang digunakan adalah:
- Accuracy
- Precision
- Recall
- F1-Score
  
A. Accuracy (Akurasi)
Akurasi mengukur proporsi prediksi yang benar terhadap seluruh data. Formula akurasi adalah:

$$
 \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Di mana:
- TP = True Positive (prediksi lulus dan benar)
- TN = True Negative (prediksi tidak lulus dan benar)
- FP = False Positive (diprediksi lulus tapi tidak lulus)
- FN = False Negative (diprediksi tidak lulus padahal lulus)

Akurasi baik digunakan jika distribusi kelas seimbang, namun bisa menyesatkan jika kelas tidak seimbang.

B. Precision
Precision mengukur berapa proporsi prediksi positif (lulus) yang benar-benar tepat.

$$
 \text{Precision} = \frac{TP}{TP + FP}
$$

Metrik ini penting ketika kesalahan false positive harus dikurangi, misalnya memprediksi seseorang lulus padahal tidak.

C. Recall (Sensitivity)
Recall menunjukkan berapa banyak dari seluruh mahasiswa yang benar-benar lulus berhasil diprediksi dengan benar.

$$
 \text{Recall} = \frac{TP}{TP + FN}
$$

Metrik ini penting untuk menghindari mahasiswa yang sebenarnya berisiko tidak lulus tapi tidak terdeteksi.

D. F1-Score
F1 Score merupakan harmonisasi antara precision dan recall.

$$
 \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Berguna saat kita ingin keseimbangan antara FP dan FN.

**Hasil Evaluasi Model**
Setelah dilakukan pelatihan dan pengujian model terhadap data, diperoleh hasil sebagai berikut:
|  **Model**  | **Accuracy** | **Precision Kelas 1** | **Recal Kelas 1**  | **F1-Score Kelas 1**  | 
|-------------|--------------|---------------|------------|---------------|
|  Decision Tree C4.5 | 	0.9499 | 	0.97 | 0.87  | 0.92  | 
|  Random Forest | 		0.9520 | 	0.97 | 0.88  | 0.92  | 
|  Support Vector Machine (SVM) | 		0.9290 | 0.90 | 	0.88 | 0.89  | 

![image](https://github.com/user-attachments/assets/36f702c2-0702-4d46-8a5b-0fcface8bcf7)


Penjelasan:
Berdasarkan hasil evaluasi dari notebook, Random Forest tetap menjadi model terbaik dengan akurasi 95.2%, precision 0.97, dan recall 0.88 untuk kelas “lulus”. Decision Tree C4.5 juga menunjukkan performa sangat baik dengan akurasi 94.9%, sementara SVM sedikit di bawah dengan akurasi 92.9%. Semua model memiliki f1-score yang tinggi, tetapi Random Forest memberikan kombinasi terbaik antara akurasi dan stabilitas klasifikasi.

**Confusion Matrix**

![image](https://github.com/user-attachments/assets/fa1510b9-bcb8-43f7-830f-cb22f1e69981)

Gambar berikut menampilkan confusion matrix dari ketiga model:
- Decision Tree C4.5: TP = 134, TN = 321, FP = 4, FN = 20
- Random Forest: TP = 135, TN = 321, FP = 4, FN = 19
- SVM: TP = 135, TN = 310, FP = 15, FN = 19

Interpretasi
- Random Forest memiliki performa terbaik secara keseluruhan. Akurasi tertinggi (95.2%), precision dan recall seimbang, serta jumlah kesalahan paling sedikit (hanya 4 false positive dan 19 false negative).
- Decision Tree C4.5 hampir setara dengan Random Forest, hanya selisih satu false negative lebih banyak.
- SVM memiliki akurasi lebih rendah (92.9%) dan jumlah false positive yang lebih tinggi (15), artinya model ini lebih sering mengira mahasiswa akan lulus padahal tidak.

**Precision, Recall, F1-Score**
- Decision Tree C4.5: Precision (kelas 1): 0.97, Recall (kelas 1): 0.87, F1-Score (kelas 1): 0.92
- Random Forest: Precision (kelas 1): 0.97, Recall (kelas 1): 0.88, F1-Score (kelas 1): 0.92
- SVM:Precision (kelas 1): 0.90, Recall (kelas 1): 0.88, F1-Score (kelas 1): 0.89
  
![image](https://github.com/user-attachments/assets/c8632880-b407-40bd-bc7b-307a2758c0bc)


**Model Terbaik: Random Forest**
Berdasarkan hasil evaluasi model yang telah dilakukan melalui metrik klasifikasi (akurasi, precision, recall, dan f1-score) serta analisis confusion matrix, model Random Forest ditetapkan sebagai model terbaik dalam proyek ini.

**Alasan Pemilihan Random Forest**

1. Kinerja Evaluasi Terbaik Secara Konsisten
Model Random Forest mencapai akurasi tertinggi sebesar 95.20%, sedikit di atas Decision Tree C4.5 (94.99%) dan jauh di atas SVM (92.90%). Selain akurasi, model ini juga menghasilkan nilai precision 0.97, recall 0.88, dan f1-score 0.92 untuk kelas “lulus”, yang mencerminkan keseimbangan yang sangat baik antara menghindari false positives dan false negatives. Ini penting dalam konteks pendidikan, karena meminimalkan kesalahan dalam memprediksi kelulusan siswa sangat krusial.

2. Minim Kesalahan Kritis
Melalui confusion matrix, Random Forest hanya menghasilkan 4 false positive (mahasiswa diprediksi lulus padahal tidak) dan 19 false negative (mahasiswa diprediksi tidak lulus padahal lulus). Angka ini sangat rendah dan lebih baik daripada SVM, yang menghasilkan 15 false positive, berpotensi memberikan harapan kelulusan yang keliru.

3. Lebih Stabil dan Tahan Terhadap Overfitting
Berbeda dengan Decision Tree tunggal yang cenderung overfitting pada data pelatihan, Random Forest merupakan model ensemble yang membentuk banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan generalisasi. Hal ini menghasilkan prediksi yang lebih stabil dan tahan terhadap fluktuasi data.

4. Kemampuan Menangani Fitur yang Beragam
Dataset yang digunakan mencakup berbagai jenis fitur, baik numerik seperti GPA dan Absences, maupun kategorikal seperti Tutoring dan ParentalEducation. Random Forest mampu mengelola kombinasi fitur ini secara efisien tanpa perlu banyak pra-proses kompleks, menjadikannya fleksibel untuk berbagai jenis input.

5. Memungkinkan Interpretasi Melalui Feature Importance
Selain akurasi, Random Forest juga memungkinkan dilakukannya analisis feature importance, yang memberikan wawasan berharga tentang faktor-faktor paling berpengaruh terhadap kelulusan. Ini sangat berguna bagi pengambil kebijakan di institusi pendidikan dalam menyusun strategi intervensi berbasis data.

**Evaluasi Terhadap Business Understanding**

✅ Menjawab Problem Statement

Bagaimana memprediksi status kelulusan mahasiswa berdasarkan data akademik dan non-akademik?
- Telah berhasil dibangun model klasifikasi menggunakan Random Forest dengan akurasi tinggi (95.2%) yang mampu memprediksi kelulusan mahasiswa (lulus atau tidak lulus) berdasarkan enam fitur input utama. Evaluasi dengan confusion matrix dan metrik klasifikasi menunjukkan bahwa model ini dapat mengidentifikasi mahasiswa berisiko dengan kesalahan minimum.

Fitur apa yang paling berpengaruh terhadap kelulusan mahasiswa?
- Berdasarkan Feature Importance dari model Random Forest (terlihat pada grafik), ditemukan bahwa fitur GPA memiliki kontribusi paling besar terhadap prediksi kelulusan, disusul oleh Absences (jumlah ketidakhadiran) dan StudyTimeWeekly (waktu belajar per minggu). Faktor lain seperti pendidikan orang tua, tutoring, dan volunteering memiliki pengaruh yang jauh lebih kecil.

Model machine learning mana yang paling akurat dan sesuai digunakan?
- Setelah dibandingkan dengan Decision Tree C4.5 dan SVM, model Random Forest dipilih sebagai yang terbaik karena konsisten unggul dalam semua metrik evaluasi utama, serta memberikan hasil yang lebih stabil dan generalisasi lebih baik.

🎯 Mencapai Goals
Membangun model prediksi kelulusan mahasiswa
- Tujuan ini telah dicapai dengan baik menggunakan model Random Forest, yang mampu memberikan prediksi akurat terhadap status kelulusan mahasiswa pada data uji.

Mengidentifikasi variabel yang paling berpengaruh
- Tujuan ini dicapai melalui analisis feature importance. Hasil menunjukkan bahwa GPA dan Absences adalah dua indikator paling penting dalam menentukan kelulusan, memberikan insight yang relevan bagi kebijakan akademik.

Membandingkan beberapa algoritma klasifikasi
- Tiga algoritma diuji dan dibandingkan secara adil menggunakan GridSearchCV. Random Forest unggul secara konsisten dan dipilih sebagai model final.

📈 Dampak dari Solution Statement

Solution statement yang diajukan sebelumnya menyebutkan bahwa proyek akan menggunakan beberapa algoritma (Decision Tree C4.5, Random Forest, dan SVM) serta melakukan hyperparameter tuning untuk meningkatkan performa model dan memilih solusi terbaik berdasarkan evaluasi metrik.

Hasilnya:

Random Forest memang menjadi solusi terbaik dengan kinerja paling optimal.
Proses GridSearchCV berhasil meningkatkan akurasi model melalui pemilihan parameter yang tepat (misalnya: n_estimators=50, max_depth=10, class_weight='balanced').
Insight dari feature importance memungkinkan stakeholder seperti dosen, kaprodi, atau biro akademik untuk lebih fokus pada peningkatan GPA mahasiswa dan menurunkan angka absensi, karena kedua faktor ini terbukti paling berpengaruh terhadap kelulusan.

Secara keseluruhan, proyek ini berhasil menjawab semua pertanyaan kunci bisnis, mencapai semua tujuan analisis, dan memberikan solusi terukur dan bisa diimplementasikan. Selain menyediakan model prediksi, proyek ini juga menawarkan insight nyata yang bisa dijadikan dasar dalam penyusunan kebijakan akademik berbasis data.

## Kesimpulan
![image](https://github.com/user-attachments/assets/f5a9c0e3-593b-4802-aa3e-0c723003e4ef)

GPA dan absensi adalah faktor paling berpengaruh dalam prediksi. Waktu belajar memiliki dampak sedang. Pendidikan orang tua, bimbingan, dan sukarela kurang signifikan.

![image](https://github.com/user-attachments/assets/86a8bd97-b0de-4c8c-83fa-38f5dd3a4c5a)

**insight:** 

- Mahasiswa dengan GPA lebih tinggi memiliki peluang lebih besar untuk lulus.
- Sebagian besar mahasiswa yang tidak lulus memiliki GPA rendah, menunjukkan bahwa GPA merupakan faktor penting dalam keberhasilan akademik.
Meningkatkan GPA melalui bimbingan belajar, mentoring, atau program peningkatan akademik bisa menjadi strategi efektif untuk meningkatkan tingkat kelulusan.

![image](https://github.com/user-attachments/assets/1291fcf5-1d82-4ed8-810b-a6cd7f5fe07b)

**insight:**

Mahasiswa yang memiliki banyak absensi cenderung memiliki tingkat kelulusan lebih rendah.
Sebagian besar mahasiswa yang lulus memiliki kehadiran yang tinggi, menunjukkan bahwa kehadiran dalam kelas berperan penting dalam keberhasilan akademik.
Jika ada mahasiswa yang banyak absen tapi tetap lulus, kemungkinan mereka memiliki cara belajar lain seperti belajar mandiri atau akses ke materi kuliah yang cukup
Menjaga disiplin kehadiran di kelas dan memastikan mahasiswa memiliki akses ke materi jika mereka terpaksa absen bisa membantu meningkatkan kelulusan.

![image](https://github.com/user-attachments/assets/5461e30e-ba90-46b4-bb2d-b0f7e8da2174)

**insight:**

Mahasiswa yang menghabiskan lebih banyak waktu belajar per minggu cenderung memiliki tingkat kelulusan lebih tinggi.
Jika ada mahasiswa yang belajar lebih banyak tetapi tetap tidak lulus, ini bisa menunjukkan bahwa kualitas belajar lebih penting daripada sekadar jumlah jam belajar.
Jika distribusinya tidak jauh berbeda, bisa jadi faktor lain lebih dominan dalam menentukan kelulusan, seperti metode belajar atau lingkungan akademik.
Mendorong mahasiswa untuk memiliki strategi belajar yang efektif, seperti belajar dalam kelompok, menggunakan teknik aktif seperti retrieval practice, dan mengelola waktu dengan baik bisa membantu meningkatkan peluang kelulusan.

**Hasil:**

Dengan mempertimbangkan performa evaluasi, kestabilan prediksi, ketahanan terhadap overfitting, dan kemampuannya dalam menangani data multivariat, Random Forest merupakan pilihan terbaik dan paling tepat untuk diterapkan dalam sistem prediksi kelulusan mahasiswa pada proyek ini. Model ini mampu menjawab kebutuhan utama dalam problem statement: meminimalkan kesalahan prediksi dan memberikan hasil yang akurat serta dapat dipercaya untuk pengambilan keputusan akademik.


