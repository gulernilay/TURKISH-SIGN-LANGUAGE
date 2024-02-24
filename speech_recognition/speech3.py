import speech_recognition
#speech_recognition kütüphanesi, birçok konuşma tanıma servisini destekleyen bir Python kütüphanesidir.
#SpeechRecognition kütüphanesi, farklı tanıma servislerini (Google, Sphinx, Bing, vs.) ve ses kaynaklarını (mikrofon, ses dosyaları) destekler. 
import pyttsx3
#bilgisayarın ses çıkışından metin tabanlı verileri sesli olarak okumanıza olanak tanır. 

# Tanıyıcı oluşturulur
recognizer = speech_recognition.Recognizer()

while True:
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        #Arka plan gürültüsünü dikkate alarak ses kaynağının gürültü seviyesini ayarlanır.
        #arka plan gürültüsünü ölçmek ve tanıma işlemi sırasında bu gürültüyü dikkate alarak daha iyi sonuçlar elde etmek için kullanılır.
        print("Lütfen bir şey söyleyin")
        audio = recognizer.listen(mic)
        #mikrofondan gelen ses kaydedilir.

        try:
            # Konuşmayı tanı
            text = recognizer.recognize_google(audio, language="tr-TR")  
            #SpeechRecognition kütüphanesi aracılığıyla Google Konuşma Tanıma servisini kullanarak sesi metne dönüştüren işlemi gerçekleştirir
            text = text.lower()
            #metin işleme ve karşılaştırmaları büyük/küçük harf duyarlı olmayan bir şekilde gerçekleştirmek için
            print(f"Tanıma Sonucu: {text}")

        except speech_recognition.UnknownValueError:
            # Konuşma tanınamadı
            continue
