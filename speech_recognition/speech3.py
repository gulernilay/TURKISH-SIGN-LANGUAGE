import speech_recognition
import pyttsx3

# Initialize the speech recognition recognizer
recognizer = speech_recognition.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    """ Function to speak the provided text. """
    engine.say(text)
    engine.runAndWait()

while True:
    with speech_recognition.Microphone() as mic:
        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        speak("Lütfen bir şey söyleyin")
        print("Dinleniyor...")  # Print statement to show when the program is listening
        audio = recognizer.listen(mic)
        
        try:
            # Recognize speech using Google's speech recognition
            text = recognizer.recognize_google(audio, language="tr-TR")
            text = text.lower()
            print(f"Tanıma Sonucu: {text}")
            speak("Anladım: " + text)
        
        except speech_recognition.UnknownValueError:
            # Handle unrecognized speech
            print("Üzgünüm, konuşmanızı anlayamadım.")
            speak("Üzgünüm, konuşmanızı anlayamadım.")
        except speech_recognition.RequestError as e:
            # Handle errors in the speech recognition request
            print("Google Speech Recognition servisine erişilemiyor; {0}".format(e))
            speak("Hizmete erişilemiyor")
        except Exception as e:
            # Handle other exceptions
            print("Bir hata oluştu: {0}".format(e))
            speak("Bir hata oluştu")

