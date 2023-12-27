import numpy as np
import tensorflow as tf
import librosa

SAMPLE_RATE = 48000
def resample_audio(path):
    audio, sr = librosa.load(path)
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (1,48000-len(audio)), "constant")
    else:
        audio = audio[:SAMPLE_RATE]
    return audio

class Audio_Classifier():
    def __init__(self):
        labels = ['Parsa','Azra Khedadmand','Davood Fazeli','Javad Nematollahi','Parisa Baqerzade','kiana jhnshid','Maryam Saeedi','Matin Ghorbani','Shima Bazzazan','Mohammad','Nima','Omid nomiri','Khadijeh Valipour','Abdollah Ramezani','Rezaie','Sajedeh Gharabadiyan','Mohammad_prf']

    def train(model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
    def predict(train_data,validation_data,model):  
        model.fit(train_data, validation_data=validation_data, epochs=30)

    def preprocess(wav_filename):
        asd = resample_audio(wav_filename)
        input_data = asd.reshape(1,48000,1) 
        return(input_data)       
  
    def postprocess(model,input_data):  
      preds = np.argmax(model(input_data))
      return(preds)

    def load_model():
        model_friends = tf.keras.models.load_model("/content/drive/MyDrive/Audio/audio_friends.h5")
        return(model_friends)

