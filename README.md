# Proje Hakkında

Bu proje, **Türkçe konuşma seslerini yazıya dönüştüren** ve yazıya dönüştürülen metni **gelişmiş Türkçe özetleme** algoritması ile özetleyen bir web uygulamasıdır. Ayrıca özetlenen metni Word veya PDF dosyası olarak indirmenize olanak sağlar.

---

## Özellikler

1. **Ses Tanıma:**
   - Wav2Vec2 modeli ile ses dosyalarından Türkçe metin elde eder.
2. **Metin Özetleme:**
   - Türkçe metinleri kelime frekansı, anahtar kelimeler, cümle pozisyonu gibi kriterlerle özetler.
3. **Metin İndirme:**
   - Özetlenen metni Word (.docx) veya PDF formatında indirebilirsiniz.
4. **Noktalama ve Düzeltme:**
   - Konuşma metnine noktalama işaretleri ekler ve eksik ilk harfleri düzeltir.
5. **Otomatik Bağımlılık Kontrolü:**
   - Program başlarken eksik Python kütüphanelerini kontrol edip, yüklemeye çalışır.
6. **Model Yükleme:**
   - Eğitilmiş Wav2Vec2 modeli ve özel Türkçe özetleyici sınıfı yüklenir.

---

## Kullanılan Teknolojiler ve Kütüphaneler

- Python 3.8+
- Flask
- PyTorch
- Transformers (Huggingface)
- torchaudio
- NLTK
- pydub
- python-docx
- reportlab
- numpy
- requests
- base64

---

## Kurulum ve Çalıştırma Adımları

1. **Projeyi Klonla:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Gerekli Paketleri Yükle:**
    - Otomatik kontrol ve yükleme için program başlangıcında gerekli kontroller yapılır.
    - Manuel yüklemek istersen:
    ```bash
    pip install -r requirements.txt
    ```

3. **Model Dosyasını Yerleştir:**
    - `trained_wav2vec_model` klasörünü proje ana dizinine koy.

4. **Uygulamayı Başlat:**
    ```bash
    python app.py
    ```

5. **Web Arayüzüne Erişim:**
    - Tarayıcınızda `http://127.0.0.1:5000` adresini açın.

---

## API Endpointleri

1. **Ana Sayfa**

    - `GET /`
    - Uygulamanın ana HTML sayfasını döner.

2. **Ses İşleme**

    - `POST /process_audio`
    - Gönderilen Base64 formatındaki ses dosyasını yazıya çevirir.
    - JSON Gönderimi:
      ```json
      {
        "audio": "data:audio/wav;base64,...."
      }
      ```
    - JSON Yanıtı:
      ```json
      {
        "success": true,
        "transcription": "Metne dönüştürülmüş konuşma",
        "timestamp": "20 June 2025, 14:00"
      }
      ```

3. **Metin Özetleme**

    - `POST /summarize`
    - Gönderilen metni istenilen cümle sayısına göre özetler.
    - JSON Gönderimi:
      ```json
      {
        "text": "Özetlenecek metin buraya",
        "num_sentences": 3
      }
      ```
    - JSON Yanıtı:
      ```json
      {
        "success": true,
        "summary": "Özetlenmiş metin",
        "original_length": 200,
        "summary_length": 60,
        "compression_ratio": 70.0,
        "num_sentences": 3
      }
      ```

4. **Dosya İndirme**

    - `POST /download/<format_type>`
    - Metni Word veya PDF formatında indirir.
    - `<format_type>`: `word` veya `pdf`
    - JSON Gönderimi:
      ```json
      {
        "text": "İndirilecek metin"
      }
      ```
    - Yanıt olarak dosya indirme başlatılır.

---

## Özetleme Algoritması

- Metni cümlelere ayırır.
- Kelime frekanslarına göre her cümle puanlanır.
- Önemli ve az önemli kelimeler dikkate alınır.
- Cümle pozisyonu, uzunluk ve teknik terimler değerlendirilir.
- En yüksek puanlı cümleler seçilerek özet oluşturulur.

---

## Font ve Dosya İşlemleri

- PDF dosyaları oluşturulurken `DejaVuSans.ttf` fontu kullanılır ve uygulama başlangıcında otomatik indirilir.
- Word ve PDF dosyaları geçici dosyalar olarak oluşturulur ve indirilebilir.

---



## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.
