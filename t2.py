import pickle
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import speech_recognition as sr

def predict():
    def preprocess_text(sen):
        sentence = remove_tags(sen)
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    TAG_RE = re.compile(r'<[^>]+>')
    def remove_tags(text):
        return TAG_RE.sub('', text)

    recognizer = sr.Recognizer()

    def get_audio():
        with sr.Microphone() as source:
            print("Say something:")
            audio = recognizer.listen(source)
            print("Time over, thanks")
        try:
            text = recognizer.recognize_google(audio)
            print("Recognized text:", text)  # Print the recognized text
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

    model = load_model('model_emotionnew.hdf5')
    with open('tokenizer_emotio.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    categories = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

    input_text = get_audio()

    preprocessed_text = preprocess_text(input_text)

    X = tokenizer.texts_to_sequences([preprocessed_text])
    X_test = pad_sequences(X, padding='post', maxlen=100)

    pred = model.predict(X_test)
    emotion_index = np.argmax(pred)
    emotion = categories[emotion_index]

    return emotion

emotion = predict()
print("Detected emotion:", emotion)
