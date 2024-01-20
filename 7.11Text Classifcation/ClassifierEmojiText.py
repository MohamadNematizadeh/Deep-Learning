import tensorflow as tf
import numpy as np
import pandas as pd

class EmojiTextClassifier():
    def __init__(self):
        pass
        
    def load_dataset(self,read_csv):
            self. X_train , self.Y_train = read_csv("/content/drive/MyDrive/dataset/Emoji_Text_Classification/train.csv")
            self.X_test , self.Y_test = read_csv("/content/drive/MyDrive/dataset/Emoji_Text_Classification/test.csv")
          
    def load_feature_vectors(self):
        self.f = open("/content/drive/MyDrive/dataset/glov.6b/glove.6B.50d.txt", encoding="utf_8")
        self.word_vectors = {}
        for line in self.f:
            line = line.strip().split()
            word = line[0]
            vector = np.array(line[1:],dtype=np.float64)
            self.word_vectors[word] = vector
        return self.word_vectors

    def sentence_to_feature_vectors_avg(self,sentence):
              self.sentence = sentence.lower()
              words = self.sentence.strip().split(' ')
              sum_vectors = np.zeros((50,))
              for word in words:
                  sum_vectors += self.word_vectors[word]

              avg_words = sum_vectors / len(words)

              return avg_words

    def load_model(self,avg_vectors=None): 
        self.model = tf.keras.models.load_model("/content/drive/MyDrive/Emoji_text_classiftcation.h5")
        return self.model

    def label_to_emoji(self,label):
        self.label = label
        emojies=['❤️','⚾','😊','😞','🍴']

        return emojies[self.label]

    def predict(self ,my_test):
        self.sentence_test = my_test
        self.my_test_avg = self.sentence_to_feature_vectors_avg(self.sentence_test)
        self.my_test_avg = np.array([self.my_test_avg])
        self.result = self.model.predict(self.my_test_avg)
        self.y_pred = np.argmax(self.result)
        emojie = self.label_to_emoji(self.y_pred)
        print(emojie)



def senteence_to_avg(sentence):
    sentence = sentence.lower()
    words = sentence.strip().split(' ')
    sum_vectors = np.zeros((50,))
    for word in words:
        sum_vectors += word_vectors[word]
    avg_words = sum_vectors / len(words)

    return avg_words