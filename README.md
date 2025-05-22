# 🏡📊 House Prices Tahmin Modeli: Kapsamlı Regresyon Analizi

---

## 1. Proje Hakkında

Bu proje, Kaggle'ın popüler **"House Prices - Advanced Regression Techniques"** veri setini kullanarak, Ames, Iowa'daki konutların çeşitli yapısal ve konumsal özelliklerine dayanarak nihai **satış fiyatlarını tahmin etmeyi** amaçlayan uçtan uca bir makine öğrenimi çalışmasıdır. Proje, veri bilimi sürecinin tüm temel adımlarını (veri toplama, temizleme, keşifçi veri analizi, özellik mühendisliği, modelleme ve değerlendirme) detaylı bir şekilde kapsar.

**Proje Hedefi:** Konut özelliklerini kullanarak, `SalePrice` (Satış Fiyatı) gibi sürekli bir değeri doğru bir şekilde tahmin eden sağlam bir regresyon modeli geliştirmek.

---

## 2. Veri Seti

Kullanılan veri seti, 79 açıklayıcı özellik (bağımsız değişken) ve 1 hedef değişken (`SalePrice`) ile Ames, Iowa'daki 2919 konut gözlemini içerir. Veri seti, sayısal, kategorik (nominal ve ordinal) ve kayıp değerler içeren zengin bir yapıya sahiptir.

* **Veri Kaynağı:** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

---

## 3. Kullanılan Araçlar ve Kütüphaneler

Proje, Python programlama dili ve aşağıdaki temel veri bilimi kütüphaneleri kullanılarak geliştirilmiştir:

* **`pandas`**: Veri manipülasyonu ve analizi için.
* **`numpy`**: Sayısal işlemler ve matematiksel fonksiyonlar için.
* **`matplotlib` & `seaborn`**: Veri görselleştirme ve keşifçi veri analizi (EDA) için.
* **`scipy.stats`**: İstatistiksel analizler ve dağılım incelemeleri için.
* **`scikit-learn`**: Makine öğrenimi modelleme, ön işleme (StandardScaler), model seçimi (train_test_split, KFold, GridSearchCV) ve değerlendirme metrikleri için.
* **`warnings`**: Uyarı mesajlarını yönetmek için.

---

## 4. Proje Akışı ve Yöntemler

Bu proje, tipik bir veri bilimi proje döngüsünü takip eder:

### 4.1. Veri Yükleme ve Ön İşleme

* **Veri Entegrasyonu:** `train.csv` ve `test.csv` dosyaları tek bir DataFrame'de birleştirilerek tutarlı bir ön işleme akışı sağlandı.
* **Eksik Değer Yönetimi:** Alan bilgisi (`data_description.txt`) kullanılarak, eksik değerlerin (örn. `PoolQC`, `Alley`, `Bsmt` ile ilgili sütunlar) 'Yokluk' anlamına geldiği durumlarda `None` veya `0` ile dolduruldu. Diğer durumlarda medyan veya mod ile doldurma stratejileri uygulandı.
* **Hedef Değişken Dönüşümü:** `SalePrice` sütunundaki **pozitif çarpıklığı gidermek** ve dağılımını normalleştirmek için `np.log1p` (**logaritmik dönüşüm**) uygulandı. Bu, regresyon modellerinin varsayımlarına daha uygun bir veri yapısı sağladı.
    ![SalePrice Dağılımı: Dönüşüm Öncesi ve Sonrası](assets/saleprice_distribution.png)
    *(Görsel Notu: Lütfen bu kısma `SalePrice`'ın log dönüşümü öncesi ve sonrası dağılımını gösteren bir görsel ekleyin.)*

### 4.2. Özellik Mühendisliği (Feature Engineering)

Mevcut özelliklerden daha anlamlı ve tahmin gücü yüksek yeni özellikler türetildi:

* **`TotalSF`**: Bodrum, 1. ve 2. kat alanlarının toplamı.
* **`TotalBath`**: Tüm tam ve yarım banyoların toplamı.
* **`YearsSinceBuilt`** / **`YearsSinceRemodel`**: Evin yaşı ve yenilenme yaşı.
* **`GarageAge`**: Garajın yaşı.
* **`OverallScore`**: Genel kalite ve kondisyon puanlarının toplamı.
* **`TotalPorchSF`**: Tüm veranda alanlarının toplamı.
* **Etkileşim Özellikleri**: `OverallQual` * `GrLivArea` ve `KitchenQual` * `GrLivArea` gibi çarpım özellikleriyle karmaşık ilişkiler yakalandı.
* **Veri Tipi Dönüşümü**: `MSSubClass`, `OverallQual`, `OverallCond` gibi sayısal görünen ancak kategorik olan sütunlar doğru şekilde `object` tipine dönüştürüldü.
* **Kategorik Özellik Dönüşümü**:
    * **Sıralı (Ordinal) Kategorikler**: `ExterQual`, `KitchenQual`, `BsmtQual` gibi kalite ve kondisyon özelliklerine `map` fonksiyonu ile manuel olarak sayısal sıralamalar atandı.
    * **Nominal (Sırasız) Kategorikler**: Geriye kalan tüm nominal kategorik özellikler için **One-Hot Encoding** (`pd.get_dummies`) uygulandı.

### 4.3. Aykırı Değer Yönetimi

* Model performansını olumsuz etkileyebilecek `GrLivArea` gibi temel özelliklerdeki aşırı aykırı değerler incelendi ve alan bilgisiyle belirlenen sınırlar dahilinde yönetildi (örn. çok büyük alanlı ve düşük fiyatlı evlerin düşürülmesi).

### 4.4. Özellik Ölçeklendirme (Feature Scaling)

* Regresyon modellerinin (özellikle Lineer Regresyon ve Ridge Regresyon) daha iyi performans göstermesi ve hızlı yakınsama sağlaması için tüm sayısal özellikler **`StandardScaler`** ile ölçeklendirildi.

### 4.5. Modelleme ve Değerlendirme

Veri setleri eğitim ve test kümelerine ayrıldıktan sonra (iç test için), çeşitli regresyon modelleri eğitildi ve performansları değerlendirildi.

* **Modeller:**
    * **Linear Regression**: Bir temel model olarak kullanıldı.
    * **Ridge Regression**: L2 regülarizasyonu ile aşırı uyumu ve multi-kolineleriteyi ele almak için uygulandı.
    * **Random Forest Regressor**: Daha karmaşık doğrusal olmayan ilişkileri yakalayabilen bir topluluk öğrenme (ensemble) modeli kullanıldı.
* **Değerlendirme Metrikleri:**
    * **MAE (Mean Absolute Error)**
    * **RMSE (Root Mean Squared Error)**
    * **R2 (R-squared)**
* **Çapraz Doğrulama (K-Fold Cross-Validation):** Modellerin genellenebilirliğini ve performans tutarlılığını daha güvenilir bir şekilde ölçmek için 5-katmanlı çapraz doğrulama uygulandı.

### 4.6. Model Performans Karşılaştırması

Test seti üzerinde elde edilen metrikler:

| Model             | MAE          | RMSE         | R2           |
| :---------------- | :----------- | :----------- | :----------- |
| Ridge Regression  | 0.0870       | 0.1329       | 0.8953       |
| Random Forest     | 0.0935       | 0.1419       | 0.8806       |
| Linear Regression | Çok Yüksek   | Çok Yüksek   | Çok Negatif  |
*(Not: Linear Regression modelinde ciddi bir sayısal istikrarsızlık veya veri uyumsuzluğu yaşanmıştır, bu nedenle değerleri anlamsızdır ve özel inceleme gerektirir.)*

**Çıkarım:** Ridge Regresyon modeli, bu veri seti üzerinde ve uygulanan ön işleme adımları ile **en iyi performansı** sergilemiştir. RMSE'si en düşük, R2 değeri en yüksek olan modeldir. Random Forest da oldukça iyi bir performans göstermekle birlikte, bu özel test seti üzerinde Ridge'in biraz gerisinde kalmıştır.

### 4.7. Hata Analizi ve Özellik Önem Derecesi

Seçilen modelin (örn. Ridge veya Random Forest) tahmin hataları incelenerek, modelin hangi durumlarda zorlandığı ve hangi özelliklerin tahmin sürecinde en önemli rol oynadığı belirlendi.
![Gerçek vs. Tahmin Edilen Fiyatlar](assets/predictions_vs_actuals.png)
*(Görsel Notu: Lütfen bu kısma en iyi modelinizin gerçek fiyatlar ile tahmin edilen fiyatları karşılaştıran bir scatter plot ekleyin.)*
![En Önemli Özellikler](assets/feature_importances.png)
*(Görsel Notu: Lütfen bu kısma Random Forest modelinden elde edilen en önemli 15 özelliği gösteren bir bar plot ekleyin.)*

---

## 5. Gelecek Çalışmalar ve İyileştirme Alanları

Bu proje sağlam bir temel sunsa da, modelin performansını daha da artırmak için çeşitli iyileştirme alanları mevcuttur:

* **Daha İleri Regresyon Modelleri:** XGBoost, LightGBM, CatBoost gibi gradient boosting algoritmaları denenmeli.
* **Hiperparametre Optimizasyonu:** `GridSearchCV` veya `RandomizedSearchCV` ile daha geniş bir hiperparametre alanı aranmalı ve modellerin ince ayarları yapılmalı.
* **Özellik Seçimi:** Modelin karmaşıklığını azaltmak ve genellenebilirliği artırmak için özellik seçimi teknikleri uygulanmalı.
* **Ensemble Yöntemleri:** Birden fazla modelin tahminleri birleştirilerek (stacking, blending) daha sağlam sonuçlar elde edilmeli.
* **Daha Detaylı Hata Analizi:** Modelin en büyük hataları yaptığı spesifik evlerin özellikleri derinlemesine incelenerek, veri setindeki eksik bilgiler veya özel durumlar tespit edilmeli.

---

## 6. Nasıl Çalıştırılır?

Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git)
    cd REPO_ADINIZ
    ```
2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Not: `requirements.txt` dosyasını `pip freeze > requirements.txt` komutu ile projenizde kullandığınız kütüphanelerin listesini çıkararak oluşturabilirsiniz.)*
3.  **Veri Setini İndirin:**
    Kaggle House Prices yarışmasından `train.csv` ve `test.csv` dosyalarını indirip projenizin ana dizinine yerleştirin.
4.  **Jupyter Notebook'u Başlatın:**
    ```bash
    jupyter notebook
    ```
    Ardından `House_Prices_Regresyon_Analizi.ipynb` (veya notebook dosyanızın adı ne ise) dosyasını açarak hücreleri sırayla çalıştırın.

---
