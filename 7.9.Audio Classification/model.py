import tensorflow as tf
class Model():
    def __init__(self,output=17,channel=32):
        super().__init__()
        self.Conv1=tf.keras.layers.Conv1D(channel, kernel_size=80, activation='relu', strides=16, input_shape=(48000, 1)),
        self.MaxPooling1=tf.keras.layers.MaxPooling1D(4),
        self.Conv2=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        self.MaxPooling2=tf.keras.layers.MaxPooling1D(4),
        self.Conv3=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        self.MaxPooling3=tf.keras.layers.MaxPooling1D(4),
        self.Conv4=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        self.MaxPooling4=tf.keras.layers.MaxPooling1D(4),
        self.Conv5=tf.keras.layers.Conv1D(channel, kernel_size=3, activation='relu'),
        self.MaxPooling5=tf.keras.layers.MaxPooling1D(4),
        self.Flatten=tf.keras.layers.Flatten(),
        self.Dense=tf.keras.layers.Dense(output, activation='softmax')
        return tf.keras.models.Sequential([self.Conv1,self.MaxPooling1,self.Conv2,self.MaxPooling2,self.Conv3,self.MaxPooling3,self.Conv4,self.MaxPooling4,self.Conv5,self.MaxPooling5,self.Flatten,self.Dense])
