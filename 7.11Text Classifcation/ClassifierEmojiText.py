import tensorflow as tf
import numpy as np
import pandas as pd

class EmojiTextClassifier():
    def __init__(self):
        pass
        
    def load_dataset(self,read_csv):
            self. X_train , self.Y_train = read_csv("/content/drive/MyDrive/dataset/Emoji_Text_Classification/train.csv")
            self.X_test , self.Y_test = read_csv("/content/drive/MyDrive/dataset/Emoji_Text_Classification/test.csv")
          
    def load_feature_vectors(self,hrf):
        self.f = open(hrf, encoding="utf_8")
        self.word_vectors = {}
        for line in self.f:
            line = line.strip().split()
            word = line[0]
            vector = np.array(line[1:],dtype=np.float64)
            self.word_vectors[word] = vector
        return self.word_vectors

    def sentence_to_feature_vectors_avg(self,sentence,veger_number):
          self.sentence = sentence.lower()
          self.words = sentence.strip().split(" ")
          self.sum_vectors = np.zeros((veger_number, ))
          for word in self.words:
            self.sum_vectors += self.word_vectors[word]
            
          self.avg_vectors = self.sum_vectors / len(self.words)
          return self.avg_vectors

    def load_model(self,hrf_model,avg_vectors=None): 
        self.model = tf.keras.models.load_model(hrf_model)
        return self.model

    def label_to_emoji(self,label):
        self.label = label
        emojies=['❤️','⚾','😊','😞','🍴']

        return emojies[self.label]

    def predict(self ,my_test,veger_number):
        self.sentence_test = my_test
        self.my_test_avg = self.sentence_to_feature_vectors_avg(self.sentence_test,veger_number)
        self.my_test_avg = np.array([self.my_test_avg])
        self.result = self.model.predict(self.my_test_avg)
        self.y_pred = np.argmax(self.result)
        emojie = self.label_to_emoji(self.y_pred)
        print(emojie)
        # return self.label_to_emoji(y_pred)

    def test(self,test=True):
        result = self.model.evaluate(self.X_test, self.Y_test)    
        return result
