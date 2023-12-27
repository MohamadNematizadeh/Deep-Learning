import numpy as np
import tensorflow as tf
# import librosa

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
        labels = ['Parsa','Azra Khedadmand','Davood Fazeli','Javad Nematollahi','Parisa Baqerzade','kiana jhnshid','Maryam Saeedi','Matin Ghorbani','Shima Bazzazan','Mohammad','Nima','Omid nomiri','Khadijeh Valipour','Abdollah Ramezani','Rezaie','Sajedeh Gharabadiyan','Mohammad_prf']

    def train(model,train_data,validation_data):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        model.fit(train_data, validation_data=validation_data, epochs=30)
        
    def predict(train_data,validation_data,model):  
        model.fit(train_data, validation_data=validation_data, epochs=30)

    def preprocess(wav_filename):
        asd = resample_audio(wav_filename)
        input_data = asd.reshape(1,48000,1) 
        return(input_data)       
  
    def postprocess(model,input_data):  
      preds = np.argmax(model(input_data))
      return(preds)
      
    def create_model(train_data,validation_data,output=17,channel=32,):
        Conv1=tf.keras.layers.Conv1D(channel, kernel_size=80, activation='relu', strides=16, input_shape=(48000, 1)),
        MaxPooling1=tf.keras.layers.MaxPooling1D(4),
        Conv2=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        MaxPooling2=tf.keras.layers.MaxPooling1D(4),
        Conv3=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        MaxPooling3=tf.keras.layers.MaxPooling1D(4),
        Conv4=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        MaxPooling4=tf.keras.layers.MaxPooling1D(4),
        Conv5=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        MaxPooling5=tf.keras.layers.MaxPooling1D(4),
        Flatten=tf.keras.layers.Flatten(),
        Dense=tf.keras.layers.Dense(output, activation='softmax')
        
    def call(self, x):
        x = tf.layers.flatten(x)
        x = self.Conv1(x)
        x = self.MaxPooling1(x)
        x = self.Conv2(x)
        x = self.MaxPooling2(x)
        x = self.Conv3(x)
        x = self.MaxPooling3(x)
        x = self.Conv4(x)
        x = self.MaxPooling4(x)
        x = self.Conv5(x)
        x = self.MaxPooling5(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        return x

        # return model

    def load_model():
        model_friends = tf.keras.models.load_model("/content/drive/MyDrive/Audio/audio_friends.h5")
        return(model_friends)

