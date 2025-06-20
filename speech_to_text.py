from flask import Flask, render_template, jsonify, request, send_file
import sounddevice as sd
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import subprocess
import pkg_resources
import os
import base64
import json
from datetime import datetime
from pydub import AudioSegment
from io import BytesIO
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import tempfile
import nltk
import re
from collections import Counter
import requests
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

app = Flask(__name__)

# Global variables for models
device = None
wav2vec_processor = None
wav2vec_model = None
summarizer_model = None
summarizer_tokenizer = None
summarizer_device = None
turkish_summarizer = None


class TurkishTextSummarizer:
    def __init__(self):
        # NLTK gerekli dosyaları indir
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Türkçe'de önemli anahtar kelimeler (yüksek puan)
        self.important_keywords = {
            'önemli', 'kritik', 'temel', 'ana', 'asıl', 'başlıca', 'öncelikli',
            'gelişmiş', 'yeni', 'modern', 'teknolojik', 'dijital', 'akıllı',
            'sistem', 'platform', 'uygulama', 'çözüm', 'yöntem', 'teknik',
            'araştırma', 'geliştirme', 'inovasyon', 'buluş', 'keşif',
            'etkili', 'verimli', 'hızlı', 'güvenli', 'kaliteli', 'profesyonel',
            'uzman', 'deneyimli', 'nitelikli', 'sertifikalı', 'onaylı',
            'başarılı', 'popüler', 'yaygın', 'genel', 'evrensel', 'standart'
        }

        # Türkçe'de az önemli kelimeler (düşük puan)
        self.unimportant_words = {
            've', 'veya', 'ile', 'için', 'gibi', 'kadar', 'göre', 'dolayı',
            'bu', 'şu', 'o', 'bir', 'birkaç', 'çok', 'az', 'daha', 'en',
            'ama', 'fakat', 'lakin', 'ancak', 'sadece', 'yalnızca',
            'de', 'da', 'ki', 'ise', 'iken', 'olsa', 'olsun'
        }

    def turkish_sentence_tokenize(self, text):
        """
        Türkçe metni cümlelere ayırır, virgülle ayrılmış cümleleri de dikkate alır
        """
        text = text.strip()
        sentence_endings = r'[.!?]+'
        sentences = re.split(sentence_endings, text)
        processed_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            comma_parts = sentence.split(',')
            if len(comma_parts) > 1:
                for i, part in enumerate(comma_parts):
                    part = part.strip()
                    if part:
                        if i > 0:
                            part = ',' + part
                        processed_sentences.append(part)
            else:
                processed_sentences.append(sentence)

        processed_sentences = [s for s in processed_sentences if s.strip()]
        return processed_sentences

    def calculate_sentence_score(self, sentence, position, total_sentences, word_freq):
        """
        Cümle için detaylı puan hesaplar
        """
        # Kelimeleri ayır ve temizle
        words = re.findall(r'\b\w+\b', sentence.lower())
        if not words:
            return 0

        score = 0

        # 1. Kelime frekansı puanı (daha sık geçen kelimeler daha önemli)
        freq_score = sum(word_freq.get(word, 0) for word in words)
        score += freq_score * 0.3

        # 2. Anahtar kelime puanı
        keyword_count = sum(1 for word in words if word in self.important_keywords)
        score += keyword_count * 2.0

        # 3. Az önemli kelime cezası
        unimportant_count = sum(1 for word in words if word in self.unimportant_words)
        score -= unimportant_count * 0.5

        # 4. Pozisyon puanı (başlangıç ve son cümleler daha önemli)
        if position == 0:  # İlk cümle
            score += 1.5
        elif position == total_sentences - 1:  # Son cümle
            score += 1.0
        elif position < total_sentences * 0.3:  # İlk %30
            score += 0.8
        elif position > total_sentences * 0.7:  # Son %30
            score += 0.6

        # 5. Uzunluk puanı (optimal uzunluk 10-25 kelime)
        word_count = len(words)
        if 10 <= word_count <= 25:
            score += 1.0
        elif 5 <= word_count < 10:
            score += 0.5
        elif word_count > 25:
            score += 0.3
        else:  # Çok kısa cümleler
            score -= 0.5

        # 6. Teknik terim bonusu (büyük harfle başlayan kelimeler)
        technical_terms = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        score += len(technical_terms) * 0.8

        return max(0, score)  # Negatif puan olmasın

    def extractive_summarize(self, text, num_sentences=3):
        """
        Gelişmiş extractive özetleme yöntemi ile metni özetler
        """
        # Metni cümlelere ayır
        sentences = self.turkish_sentence_tokenize(text)

        if len(sentences) <= num_sentences:
            return ' '.join(sentences)

        # Tüm metindeki kelime frekansını hesapla
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            all_words.extend(words)

        word_freq = Counter(all_words)

        # Her cümle için puan hesapla
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self.calculate_sentence_score(sentence, i, len(sentences), word_freq)
            sentence_scores.append((score, i, sentence))

        # Puanlara göre sırala (yüksek puanlılar önce)
        sentence_scores.sort(reverse=True)

        # En yüksek puanlı cümleleri seç
        top_sentences = sentence_scores[:num_sentences]

        # Orijinal sıraya göre sırala
        top_sentences.sort(key=lambda x: x[1])

        # Seçilen cümleleri birleştir
        summary = ' '.join([sentence for _, _, sentence in top_sentences])

        return summary

    def get_sentence_analysis(self, text):
        """
        Cümle analizi için detaylı bilgi döndürür
        """
        sentences = self.turkish_sentence_tokenize(text)

        if not sentences:
            return []

        # Kelime frekansını hesapla
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            all_words.extend(words)

        word_freq = Counter(all_words)

        # Her cümle için detaylı analiz
        analysis = []
        for i, sentence in enumerate(sentences):
            score = self.calculate_sentence_score(sentence, i, len(sentences), word_freq)
            words = re.findall(r'\b\w+\b', sentence.lower())

            analysis.append({
                'sentence': sentence,
                'position': i + 1,
                'score': round(score, 2),
                'word_count': len(words),
                'keywords': [w for w in words if w in self.important_keywords],
                'unimportant_words': [w for w in words if w in self.unimportant_words]
            })

        return analysis

    def summarize_with_analysis(self, text, num_sentences=3):
        """
        Özetleme ve analizi birlikte döndürür
        """
        summary = self.extractive_summarize(text, num_sentences)
        analysis = self.get_sentence_analysis(text)

        return {
            'summary': summary,
            'analysis': analysis,
            'total_sentences': len(analysis)
        }


def check_dependencies():
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'protobuf': 'protobuf',
        'sentencepiece': 'sentencepiece',
        'sounddevice': 'sounddevice',
        'numpy': 'numpy',
        'flask': 'flask',
        'python-docx': 'docx',
        'reportlab': 'reportlab',
        'nltk': 'nltk'
    }

    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            pkg_resources.require(package)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)

    if missing_packages:
        print("Eksik kütüphaneler tespit edildi. Yükleniyor...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("Kütüphaneler başarıyla yüklendi!")
        except subprocess.CalledProcessError as e:
            print(f"Kütüphane yükleme hatası: {str(e)}")
            print(f"\nLütfen bu komutu çalıştırın:\npip install {' '.join(missing_packages)}")
            return False
    return True


def load_models():
    global device, wav2vec_processor, wav2vec_model, summarizer_model, summarizer_tokenizer, summarizer_device, turkish_summarizer

    try:
        print("Wav2Vec modeli yükleniyor...")
        save_path = "trained_wav2vec_model"
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(save_path)
        wav2vec_model = Wav2Vec2ForCTC.from_pretrained(save_path, trust_remote_code=True)
        wav2vec_model.eval()
        print("Wav2Vec modeli başarıyla yüklendi!")

        # Türkçe özetleyici oluştur
        print("Türkçe özetleyici başlatılıyor...")
        turkish_summarizer = TurkishTextSummarizer()
        print("Türkçe özetleyici başarıyla yüklendi!")

        return True
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {str(e)}")
        return False


def summarize_text(text, num_sentences=3):
    """Gelişmiş Türkçe özetleme - yeni sistem"""
    global turkish_summarizer

    try:
        if turkish_summarizer is None:
            return None

        # Metni temizle
        text = text.strip()
        if not text:
            return None

        # Gelişmiş özetleme yap
        summary = turkish_summarizer.extractive_summarize(text, num_sentences)
        return summary

    except Exception as e:
        print(f"Metin özetlenirken hata oluştu: {str(e)}")
        return None


def remove_duplicate_sentences(text):
    """Aynı cümleleri tekrar etmemesi için kaldırır"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    unique_sentences = list(dict.fromkeys(sentences))
    return '. '.join(unique_sentences) + '.'


def preprocess_text(text):
    """Virgülle ayrılmışsa cümleleri noktaya çevirir ve tekrarları temizler"""
    if ',' in text and '.' not in text:
        text = '. '.join([s.strip() for s in text.split(',')]) + '.'
    return remove_duplicate_sentences(text)


def create_word_document(text, filename):
    """Word belgesi oluşturur"""
    try:
        doc = Document()
        doc.add_paragraph(text)

        # Geçici dosya oluştur
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
        doc.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        raise Exception(f"Word belgesi oluşturulurken hata: {str(e)}")


def download_dejavu_font(font_path="DejaVuSans.ttf"):
    url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        print("DejaVuSans.ttf fontu indiriliyor...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(font_path, "wb") as f:
                f.write(response.content)
            print("DejaVuSans.ttf başarıyla indirildi.")
        else:
            print("Font indirilemedi! Lütfen internet bağlantınızı kontrol edin.")
    else:
        print("DejaVuSans.ttf zaten mevcut.")


def create_pdf_document(text, filename):
    """PDF belgesi oluşturur"""
    try:
        # Fontu indir ve kaydet
        download_dejavu_font()
        # Geçici dosya oluştur
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')

        # PDF oluştur
        pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
        c = canvas.Canvas(temp_file.name, pagesize=A4)
        width, height = A4

        # Metin
        c.setFont("DejaVu", 12)
        y_position = height - 50
        line_height = 15

        # Metni satırlara böl
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) * 7 < width - 100:  # Yaklaşık karakter genişliği
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Satırları yazdır
        for line in lines:
            if y_position < 50:  # Sayfa sonu kontrolü
                c.showPage()
                y_position = height - 50
                c.setFont("DejaVu", 12)

            c.drawString(50, y_position, line)
            y_position -= line_height

        c.save()
        return temp_file.name
    except Exception as e:
        raise Exception(f"PDF belgesi oluşturulurken hata: {str(e)}")


@app.route('/')
def index():
    return render_template('index!_2.html')


@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        audio_data = request.json['audio']
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        audio_buffer = BytesIO(audio_bytes)

        # Önce WAV olarak dene, hata olursa WEBM olarak dene
        try:
            audio = AudioSegment.from_file(audio_buffer, format="wav")
        except Exception:
            audio_buffer.seek(0)
            audio = AudioSegment.from_file(audio_buffer, format="webm")

        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / (2 ** 15)

        # Model için işleme
        inputs = wav2vec_processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = wav2vec_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = wav2vec_processor.batch_decode(predicted_ids)[0]

        # <unk> etiketini temizle
        transcription = transcription.replace("<unk>", "").strip()

        # Noktalama işaretlerini ekle
        transcription = add_punctuation(transcription)

        # İlk harfi eksik kelimeleri düzelt
        transcription = fix_first_letter_missing(transcription)

        return jsonify({
            'success': True,
            'transcription': transcription,
            'timestamp': datetime.now().strftime("%d %B %Y, %H:%M")
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/summarize', methods=['POST'])
def summarize_text_endpoint():
    try:
        data = request.json
        text = data.get('text', '')
        num_sentences = data.get('num_sentences', 3)  # Yeni parametre

        if not text.strip():
            return jsonify({
                'success': False,
                'error': 'Özetlenecek metin bulunamadı'
            })

        print(f"Özetleme isteği alındı. Metin uzunluğu: {len(text)} karakter")

        # Metni özetle - yeni sistem
        summary = summarize_text(text, num_sentences)

        if summary:
            return jsonify({
                'success': True,
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': round((1 - len(summary) / len(text)) * 100, 1),
                'num_sentences': num_sentences
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Özetleme işlemi başarısız oldu'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/download/<format_type>', methods=['POST'])
def download_file(format_type):
    try:
        data = request.json
        text = data.get('text', '')

        if not text.strip():
            return jsonify({
                'success': False,
                'error': 'İndirilecek metin bulunamadı'
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type == 'word':
            filename = f"konusma_metni_{timestamp}.docx"
            file_path = create_word_document(text, filename)
            mimetype = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif format_type == 'pdf':
            filename = f"konusma_metni_{timestamp}.pdf"
            file_path = create_pdf_document(text, filename)
            mimetype = 'application/pdf'
        else:
            return jsonify({
                'success': False,
                'error': 'Desteklenmeyen format'
            })

        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def fix_first_letter_missing(sentence):
    words = sentence.split()
    fixed_words = []
    for word in words:
        # Eğer kelime sözlükte varsa düzelt
        if word in first_letter_missing_dict:
            fixed_words.append(first_letter_missing_dict[word])
        else:
            fixed_words.append(word)
    return ' '.join(fixed_words)


# Sık kullanılan Türkçe kelimelerden oluşan geniş liste
common_words = [
    "ben", "sen", "o", "biz", "siz", "onlar", "bunu", "şunu", "onu", "bana", "sana", "ona", "bize", "size", "onlara",
    "ve", "ama", "fakat", "ancak", "çünkü", "veya", "ya", "hem", "ne", "de", "da", "ki", "ile", "gibi", "için", "kadar",
    "gelmek", "gitmek", "yapmak", "etmek", "almak", "vermek", "olmak", "bulmak", "istemek", "bilmek", "görmek",
    "anlamak",
    "söylemek", "konuşmak", "yazmak", "okumak", "çalışmak", "başlamak", "bitmek", "beklemek", "sevmek", "istemek",
    "gün", "saat", "dakika", "yıl", "ay", "hafta", "bugün", "yarın", "dün", "elma", "araba", "okul", "ev", "iş",
    "kitap",
    "masa", "kalem", "telefon", "bilgisayar", "çanta", "su", "yemek", "para", "şehir", "ülke", "insan", "çocuk", "aile",
    "büyük", "küçük", "uzun", "kısa", "iyi", "kötü", "güzel", "çirkin", "yeni", "eski", "hızlı", "yavaş", "zor",
    "kolay",
    "evet", "hayır", "tamam", "merhaba", "teşekkürler", "lütfen", "görüşürüz", "hoşça", "kal", "sağol", "selam",
    "hoşgeldin","sabah","bugün"
    "hoşbulduk", "güle", "güle", "buyurun", "afiyet", "olsun", "geçmiş", "olsun", "tebrikler", "başarılar", "mutlu",
    "doğum",
    "günü", "kutlu", "olsun", "iyi", "geceler", "günaydın", "iyi", "akşamlar", "iyi", "tatiller", "iyi", "yolculuklar",
    "çok", "az", "daha", "en", "her", "bazı", "hiç", "hep", "bazen", "şimdi", "sonra", "önce", "artık", "hala", "henüz",
    "burada", "orada", "şurada", "içinde", "dışında", "üstünde", "altında", "yanında", "karşısında", "arasında",
    "ev", "okul", "iş", "yol", "hava", "su", "ateş", "toprak", "gökyüzü", "deniz", "dağ", "orman", "şehir", "köy",
    "mahalle"
]

# Sözlük: ilk harfi eksik -> tam kelime
first_letter_missing_dict = {w[1:]: w for w in common_words if len(w) > 2}


def add_punctuation(text):
    """Konuşma metnine noktalama işaretleri ekler"""
    if not text.strip():
        return text

    # Metni kelimelere ayır
    words = text.split()
    if not words:
        return text

    # Türkçe'de cümle sonu olabilecek kelimeler ve durumlar
    sentence_end_patterns = [
        # Fiil çekimleri
        r'.*dır$', r'.*dir$', r'.*dur$', r'.*dür$', r'.*tır$', r'.*tir$', r'.*tur$', r'.*tür$',
        r'.*mış$', r'.*miş$', r'.*muş$', r'.*müş$', r'.*yor$', r'.*ıyor$', r'.*iyor$', r'.*uyor$', r'.*üyor$',
        r'.*acak$', r'.*ecek$', r'.*malı$', r'.*meli$', r'.*sın$', r'.*sin$', r'.*sun$', r'.*sün$',
        # Onay/red kelimeleri
        'evet', 'hayır', 'tamam', 'tamamdır', 'anladım', 'anlıyorum', 'biliyorum', 'görüyorum',
        'düşünüyorum', 'sanıyorum', 'zannederim', 'galiba', 'belki', 'muhtemelen', 'kesinlikle',
        'kesin', 'elbette', 'tabii', 'tabii ki', 'yok', 'var', 'olur', 'olmaz', 'doğru', 'yanlış'
    ]

    # Virgül gerektiren bağlaçlar
    comma_conjunctions = {
        'ama', 'fakat', 'lakin', 'ancak', 'sadece', 'yalnızca', 'çünkü', 'zira',
        'madem', 'mademki', 'eğer', 'şayet', 've', 'veya', 'ya da', 'hem', 'hem de',
        'ne', 'ne de', 'gibi', 'kadar', 'göre', 'dolayı', 'için', 'ile', 'birlikte',
        'sonra', 'önce', 'şimdi', 'şu anda', 'bu arada', 'ayrıca', 'bunun yanında'
    }

    result = []
    i = 0

    while i < len(words):
        word = words[i].lower()
        original_word = words[i]

        # Cümle sonu kontrolü
        is_sentence_end = False

        # Pattern kontrolü
        for pattern in sentence_end_patterns:
            if isinstance(pattern, str):
                if word == pattern:
                    is_sentence_end = True
                    break
            else:  # regex pattern
                if re.match(pattern, word):
                    is_sentence_end = True
                    break

        # Özel durumlar
        if (word in ['evet', 'hayır', 'tamam', 'anladım'] and
                i + 1 < len(words) and
                words[i + 1].lower() not in comma_conjunctions):
            is_sentence_end = True

        if is_sentence_end:
            # Eğer bu kelimeden sonra bağlaç varsa virgül ekle
            if i + 1 < len(words):
                next_word = words[i + 1].lower()
                if next_word in comma_conjunctions:
                    result.append(original_word + ',')
                else:
                    result.append(original_word + '.')
            else:
                # Cümlenin sonu
                result.append(original_word + '.')

        # Virgül kontrolü - bağlaçlar
        elif word in comma_conjunctions and i > 0:
            # Önceki kelimeye virgül ekle
            if result and not result[-1].endswith(','):
                result[-1] = result[-1] + ','
            result.append(original_word)

        # Normal kelime
        else:
            result.append(original_word)

        i += 1

    # Son kelimeyi kontrol et
    if result and not result[-1].endswith(('.', '!', '?')):
        result[-1] = result[-1] + '.'

    # Metni birleştir ve temizle
    final_text = ' '.join(result)

    # Çoklu noktalama işaretlerini temizle
    final_text = re.sub(r'[.!?]+', '.', final_text)  # Birden fazla nokta varsa tek nokta yap
    final_text = re.sub(r'[,]+', ',', final_text)  # Birden fazla virgül varsa tek virgül yap
    final_text = re.sub(r'\s+', ' ', final_text)  # Fazla boşlukları temizle
    final_text = re.sub(r'\s+([,.!?])', r'\1', final_text)  # Noktalama işaretlerinden önce boşluk varsa kaldır

    return final_text.strip()


if __name__ == '__main__':
    download_dejavu_font()
    if not check_dependencies():
        print("Program başlatılamadı. Kütüphane hatası.")
        sys.exit(1)

    if not load_models():
        print("Program başlatılamadı. Model yükleme hatası.")
        sys.exit(1)

    app.run(debug=True)