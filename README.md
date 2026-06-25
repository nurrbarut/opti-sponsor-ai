# 🎯 OptiSponsorAI: Çoklu Platform Sponsorluk Bütçesinin Yapay Zeka ve Karar Analitiği ile Optimizasyonu 

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost-orange.svg)
![Optimization](https://img.shields.io/badge/Optimization-Knapsack%20Algorithm-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

**OptiSponsorAI**; Instagram, TikTok ve YouTube platformlarındaki influencer etkileşimlerini makine öğrenmesiyle tahmin ederek, işletmelerin sınırlı sponsorluk/reklam bütçelerini yöneylem araştırmasında yaygın olarak kullanılan Knapsack modelleriyle en yüksek verime (ROI) ulaştıracak şekilde planlayan uçtan uca bir karar analitiği ve optimizasyon sistemidir.

---

## 📖 Proje Genel Bakışı (Overview)
Dijital pazarlama dünyasında markaların en kritik kararlarından biri, sınırlı sponsorluk bütçelerini farklı sosyal medya platformları ve influencer'lar arasında verimli bir şekilde dağıtmaktır. Çoğu zaman bu süreç, veri analitiğine dayanmayan sezgisel yöntemlerle veya esnek olmayan geleneksel bütçe kotalarıyla yönetilmekte; bu durum reklam bütçelerinde ciddi israflara ve düşük kampanya yatırım getirisine (ROI) neden olmaktadır.

**OptiSponsorAI**, bu karar alma sürecini bilimsel kısıtlar ve tahmin modelleriyle modernize etmeyi hedefler. Sistem, influencer etkileşim potansiyellerini makine öğrenmesiyle tahmin ettikten sonra, yöneylem araştırması kısıtları altında markaya en yüksek etkileşim hacmini getirecek optimal portföyü seçer.

---

## 🛠️ Uçtan Uca Sistem Metodolojisi

### 1. Veri Harmanlama (Data Fusion)
Çok kanallı pazarlama stratejisini gerçekçi verilerle modellemek adına projede iki farklı veri katmanı entegre edilmiştir:
* **Mikro Sosyal Medya Katmanı:** Instagram, TikTok ve YouTube mecralarında faaliyet gösteren 3,000 influencer'ın takipçi, izlenme, beğeni ve kategori bazlı yapılandırılmış verileri.
* **Makro Demografik Katman:** Küresel ölçekte ülkelerin internet erişim ve kullanım oranlarını içeren veri seti (`internet-users-by-country-2024.csv`).
* *Bütünleştirme:* Bu iki katman 'hedef kitle konumu' (`country`) üzerinden birleştirilmiş, TikToker'ların konum verilerindeki boşluklar, genel dağılımın medyan değeri (%78.80) ile doldurularak veri bütünlüğü sağlanmıştır.

### 2. Hibrit Yapay Zeka ile Veri Zenginleştirme & NLP
* **Yapay Zeka Destekli Sınıflandırma:** TikTok veri setindeki eksik kategoriler, Google Gemini API ve web arama algoritmalarını kullanan hibrit bir sınıflandırma betiği (`gemini_label_tiktok.py`) vasıtasıyla zenginleştirilmiştir.
* **Semantik Metin Vektörleştirme (TF-IDF):** İçerik kategorileri, TF-IDF kelime frekansı analizine tabi tutularak modele anlamlı sayısal girdiler olarak aktarılmıştır.

### 3. İş Mantığına Dayalı Özellik Mühendisliği (Feature Engineering)
Karar mekanizmasının hassasiyetini artırmak için işletme mantığıyla 9 analitik değişken türetilmiştir. En kritik olanları:
* **`reach_index` (Erişim Endeksi):** Takipçi sayısının hedef kitle ülkesindeki internet erişim oranıyla ölçeklendirilmesiyle elde edilen net erişim gücü metriği.
* **`bot_trust_score` (Bot Güven Skoru):** Instagram etkileşimlerinin doğallık kalitesini ölçen güvenilirlik oranı.
* **`is_suspicious_bot` (Şüpheli Hesap Filtresi):** Anormal etkileşim oranlarına sahip veya pasif hesapları eleyen filtreleme bayrağı.
* **`cost_usd` (Tahmini Kampanya Maliyeti):** Sektörel CPM modelleri temel alınarak platform bazlı (izlenme/takipçi) hesaplanan gerçekçi reklam bütçesi.
* **`roi_score` (ROI Katsayısı):** Influencer'ların getireceği tahmini beğeninin, bütçe kısıtları ve platform engelleriyle dengelenerek hesaplanan birim verimlilik metriği.

---

## 📈 Makine Öğrenmesi Modelleri ve Tahmin Performansı
Sosyal medya beğenilerinin yüksek varyanslı ve çarpık dağılımını normalize etmek için model logaritmik ölçekte (`log1p`) eğitilmiştir. 3 farklı algoritmanın karşılaştırılması sonucunda Random Forest en yüksek performansı göstermiştir:

| Model | $R^2$ Skoru (Açıklayıcılık Oranı) | Mean Absolute Error (MAE) | Tahmin Doğruluğu (Log-scale)* |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | **%45.40** | **273,720 Beğeni** | **%93.19** |
| Gradient Boosting Regressor | %33.58 | 279,856 Beğeni | %92.48 |
| Optimized XGBoost | %21.91 | 288,047 Beğeni | %91.80 |

*\*Tahmin Doğruluğu, modelin logaritmik ölçekteki tahmin hatasından (MAPE) arındırılmış net doğruluğunu ($100 - \text{MAPE}$) temsil eder.*

---

## 🧠 Açıklanabilir Yapay Zeka (XAI)
Yapay zekanın karar sürecini şeffaflaştırmak amacıyla **SHAP (TreeExplainer)** kütüphanesi kullanılmıştır. Analiz sonuçlarına göre model tahminlerinde en belirleyici faktörler sırasıyla:
1. `platform_YouTube` (Platformun YouTube olması)
2. `InternetUsers_PctOfPopulationUsingInternet` (Ülkedeki internet kullanım yüzdesi)
3. `reach_index` (Türettiğimiz Erişim Endeksi değişkeni)

---

## 🎒 Bütçe Dağılım ve Optimizasyon Simülasyonu
Algoritma, bütçeyi riske atmamak (tek platforma yığılmayı önlemek) için her platforma **maksimum %50 bütçe sınırı** koyarak (Strateji 2) çalışır.

**$100,000 Bütçe Simülasyon Kıyaslaması:**
* **Strateji 1 (Sabit Oranlı - %40 / %35 / %25):** Harcanan: $83,332 | Kalan: $16,668 (%16.7 Bütçe İsrafı) | **Toplam Beğeni: 3,470,459**
* **Strateji 2 (Esnek Capped - Önerilen):** Harcanan: $99,530 | Kalan: $470 (%0.5 Bütçe İsrafı) | **Toplam Beğeni: 4,675,339** *(Katı dağılıma göre %34 verim artışı!)*
* **Strateji 3 (Sınırsız Global):** Harcanan: $99,789 | Kalan: $210 (%0.2 Bütçe İsrafı) | **Toplam Beğeni: 6,392,256** *(Bütçenin %99'u tek platforma yığıldığı için yüksek riskli.)*

---

## 💻 İnteraktif Karar Destek Arayüzü (GUI)
Notebook içerisinde yer alan `ipywidgets` tabanlı canlı simülatör sayesinde:
1. Hedef pazar sektörü açılır listeden seçilir.
2. Kampanya bütçesi kaydırma çubuğuyla belirlenir.
3. **"Optimizasyonu Çalıştır"** butonuna basıldığında model çalışır ve anlık bütçe analizi, harcanan/kalan para, tahmini toplam beğeni, **CPL (Beğeni başına maliyet)** ve **ROI (Beğeni / $1)** metriklerini hesaplar.
4. Çıktının altında açılan **katlanabilir sekmeler (Accordion)** içerisindeki HTML tablolarında seçilen tüm influencer'ların detaylı listesi sunulur.

---

## 📁 Proje Dizin Yapısı
```text
OptiSponsorAI/
│
├── OptiSponsorAI.ipynb                   # Ana veri bilimi ve optimizasyon notebook'u
├── gemini_label_tiktok.py                # Google Gemini API kategorilendirme aracı
├── internet-users-by-country-2024.csv    # Küresel internet penetrasyon veri seti
└── social_media_dataset.csv              # Ham sosyal medya veri seti
```

---
**Geliştirici:** Beyza Nur Barut
