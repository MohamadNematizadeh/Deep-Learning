import telebot
from telebot import types
import tensorflow as tf
import cv2
import numpy as np

mybot = telebot.TeleBot(token="token")
model = tf.keras.models.load_model('weights/17_Flowers.h5')
flowers_name = ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil','daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']
@mybot.message_handler(commands=['start'])
def send_welcome(message): 
    msg = mybot.send_message(message.chat.id,"سلام "+str(message.chat.first_name)+" به بات تشخیص گل خوش آمدی"+" \n"+
                            "/photo- حدست گل ")
    
@mybot.message_handler(commands=['photo'])
def send_photo(message): 
    msg = mybot.send_message(message.chat.id,"عکس یه گل بفرست(به غیر از خودت) تا اسمش رو حدس بزنم 🌹")
    mybot.register_next_step_handler(msg,photo)
	
@mybot.message_handler(content_types = ["photo"])
def photo(message):
    fileID = message.photo[-1].file_id
    file_info = mybot.get_file(fileID)
    downloaded_file = mybot.download_file(file_info.file_path)

    with open("Flowers.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    img = cv2.imread("Flowers.jpg")
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(image ,(224,224))
    img = img / 255
    img = img.reshape(1,224,224,3)        
    result = np.argmax(model.predict(img))
    print(result)
    print(flowers_name[result])

    mybot.send_message(message.chat.id,f' احساس میکنم اسم گلت {flowers_name[result]} باشه ☘️🪴 .')
mybot.infinity_polling()
