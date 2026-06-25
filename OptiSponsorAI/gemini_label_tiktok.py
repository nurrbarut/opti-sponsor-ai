import pandas as pd
import google.generativeai as genai
import json
import time
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# === YAPAY ZEKA DESTEKLİ VERİ ETİKETLEME (LLM-ASSISTED DATA ENRICHMENT) ===

print("Yapay Zeka Destekli TikTok Etiketleme Scripti Başlatılıyor...\n")

# Gemini API Anahtarını Kullanıcıdan Alıyoruz
api_key = input("Lütfen Google AI Studio Gemini API Anahtarınızı girin: ").strip()
if not api_key:
    print("API Anahtarı girilmedi. İşlem iptal edildi.")
    sys.exit(1)

# Gemini API'sini Yapılandırıyoruz
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Hedef Kategorilerimiz
target_categories = [
    'Gaming & Technology',
    'Fashion & Beauty',
    'Music & Art',
    'Sports & Fitness',
    'Entertainment & Cinema',
    'Lifestyle & Travel',
    'Other / Unknown'
]

# TikTok Veri Setini Yüklüyoruz
tiktok_path = r"C:\Users\NUR\Desktop\social media influencers - tiktok.csv"
if not os.path.exists(tiktok_path):
    print(f"Hata: Veri seti bulunamadı -> {tiktok_path}")
    sys.exit(1)

df = pd.read_csv(tiktok_path)
print(f"Toplam {len(df)} adet TikToker okunuyor...")

# API isteklerini hızlandırmak ve maliyeti düşürmek için 50'şerli paketler (Batching) halinde gönderiyoruz
batch_size = 50
labeled_data = {}

for idx in range(0, len(df), batch_size):
    batch = df.iloc[idx : idx + batch_size]
    creators = []
    for _, row in batch.iterrows():
        creators.append(f"Username: {row['Tiktoker name']} (Display Name: {row['Tiktok name']})")
        
    print(f"  -> {idx + len(batch)} / {len(df)} arası etiketleniyor, Gemini API çağrılıyor...")
    
    # Yapay zekaya göndereceğimiz prompt
    prompt = f"""
    Aşağıdaki {len(batch)} adet TikTok yayıncısını ana içerik temalarına göre şu kategorilerden birine sınıflandır:
    {target_categories}
    
    Yayıncılar:
    {chr(10).join(creators)}
    
    Yanıtı SADECE ve SADECE geçerli bir JSON objesi olarak dön. Başka hiçbir açıklama metni ekleme. JSON dışında kod blokları (```json vb.) ekleme.
    JSON formatı:
    {{
        "username_1": "Kategori",
        "username_2": "Kategori"
    }}
    buradaki anahtarlar 'Tiktoker name' (yani username) olmalıdır.
    """
    
    # API çağrısı ve hata yönetimi
    try:
        response = model.generate_content(prompt)
        # JSON yanıtı temizleme ve yükleme
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        batch_labels = json.loads(clean_text)
        
        # Etiketleri ana sözlüğe ekle (username'leri küçük harfe çekerek)
        for user, cat in batch_labels.items():
            labeled_data[user.lower().strip()] = cat
            
    except Exception as e:
        print(f"     Hata oluştu, bu paket atlanıyor: {e}")
        time.sleep(2) # Rate limit aşımını önlemek için bekleme
        continue
        
    time.sleep(1.5) # rate limit aşımı koruması

# Veri setini güncelleme
def update_category(row):
    user = str(row['Tiktoker name']).lower().strip()
    if user in labeled_data:
        cat = labeled_data[user]
        # Eğer yapay zeka listemiz dışı bir kategori döndüyse en yakına eşle
        if cat in target_categories:
            return cat
    # Varsayılan
    return 'Other / Unknown'

df['category'] = df.apply(update_category, axis=1)

# Güncellenmiş CSV'yi kaydet
df.to_csv(tiktok_path, index=False)

print("\n🚀 Tüm TikTokerlar Gemini AI tarafından otomatik olarak etiketlendi ve kaydedildi!")
print("Yeni Kategori Dağılımı:")
print(df['category'].value_counts())
