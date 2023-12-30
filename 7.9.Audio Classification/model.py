import tensorflow as tf

class M4(tf.keras.Model):
  def __init__(self):
    super(M4, self).__init__()
    Conv1=tf.keras.layers.Conv1D(32, kernel_size=80, activation='relu', strides=16, input_shape=(48000, 1)),
    MaxPooling1=tf.keras.layers.MaxPooling1D(4),
    Conv2=tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling2=tf.keras.layers.MaxPooling1D(4),
    Conv3=tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling3=tf.keras.layers.MaxPooling1D(4),
    Conv4=tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling4=tf.keras.layers.MaxPooling1D(4),
    Conv5=tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling5=tf.keras.layers.MaxPooling1D(4),
    Flatten=tf.keras.layers.Flatten(),
    Dense=tf.keras.layers.Dense(17, activation='softmax')
        
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

model = M4()
optimizer=tf.keras.optimizers.Adam()
loss='categorical_crossentropy'
model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )