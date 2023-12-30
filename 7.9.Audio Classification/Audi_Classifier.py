import numpy as np
import tensorflow as tf
import librosa
from model import M4

SAMPLE_RATE = 48000
def resample_audio(path):
    audio, sr = librosa.load(path)
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (1,48000-len(audio)), "constant")
    else:
        audio = audio[:SAMPLE_RATE]
    return audio

class Audio_Classifier(tf.keras.Model):
    def __init__(self):
        labels = ['abdollah', 'azra', 'davood', 'javad', 'kiana', 'matin', 'mohamad', 'mohamadd', 'mona', 'nima', 'omid', 'parisa', 'parsa', 'saeedi', 'sajedeh', 'shima', 'tara', 'valipour']
    def train(train_data,validation_data):
        model = M4
        model.fit(train_data, validation_data=validation_data, epochs=30)
        
    def preprocess(wav_filename):
        asd = resample_audio(wav_filename)
        desired_length = 48000
        resized_waveform = librosa.util.fix_length(asd, size = desired_length)

        input_data = np.expand_dims(resized_waveform, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)
        return(input_data)       
  
    def postprocess(model,input_data):  
      preds = np.argmax(model(input_data))
      return(preds)
    
    def create_model():
        model = M4
        print(model)
        
    def load_model():
        model_friends = tf.keras.models.load_model("/content/drive/MyDrive/Audio/audio_friends.h5")
        return(model_friends)
if __name__ == "__main__":
    audio_classify = Audio_Classifier()
    audio_classify.create_model()
