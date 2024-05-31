# 🍓 Deep Learning 🍓

## Face Recognition
Face Recognition using DeepFace library
loss and accuracy test.
  |loss | accuracy |
  | --------- |:---:|
  |0.0090 | 1.0000 |
-------------------------------------------------
 
## MLP vs CNN:

Mnist, Fashion MNist, Cifar 10, and Cifar 100.

| Benchmark Name | MLP (Machine Learning)| CNN + MLP (Deep Learning) |
| :---         |     :---:      |          :---: |
| Mnist  | 0.9719   | 0.9872   |
| Fashion Mnist     | 0.8392    |0.9133  |
|Cifar 10     |   0.4197  |   0.6958  |
|Cifar 100     | 0.1898    | 0.3395   |

-------------------------------------------------
# CustomDatasetCNN
# Transfer Learning
# 5Animals_V3
- Classification between 4 animals:
   Elephant 🐘
    dog 🐶
    cat 🐈 
    Giraffe 🦒
    Pandas 🐼
- Collect more than 200 photo data from each animal for Model training
- Using Augmentation To increase Train data

- Download the  dataset from link below:
  - [dataset](https://drive.google.com/drive/folders/1wBUlG3P8YBiB17aUo3O6byzO0DJLcsaS)
- | Dataset |  Loss	| Accuracy 
    | :---   |   :---:  | :---:   |
    |VGG16   |  0.4215 |  0.8988 |
    |Train   |  0.3229   | 0.9453  |
    |Val     |  1.2178   | 0.6771  |
    |test    |  1.3104 |  0.8929  |
-------
# 17 Flowers V3🌹🪴
- A deep learning model using VGG16 convolution neural net is trained to classify flowers 🌹🪴

- Download the  dataset from link below:
  - [dataset](https://drive.google.com/drive/folders/15Wr7hNPvFhnpkYdTpypGFmq6mcQ6r-Sx)

- | Dataset |  Loss	| Accuracy 
    | :---         |     :---:   |          :---: |
    |VGG16   |  0.4709 |  0.9188 |
    |Train  | 0.1701  | 0.9453  |
    |Val     | 1.3862    | 0.6497   |
    |test     |   1.7977 |  0.6265  |

--------------
# 7-7 faces  v2🧔🏻‍♂️👩🏽‍🦱👵🏻👨🏿‍🦲
- A deep learning model using VGG16 convolution neural net is trained to classify 15 faces🧔🏻‍♂️👩🏽‍🦱👵🏻👨🏿‍🦲

- Download the  dataset from link below:
  - [dataset](https://drive.google.com/drive/folders/1WGSotRtFPYGuxPEGkWWRsBPlVXFSvl7p?usp=sharing)
  
- | Dataset |  Loss	| Accuracy 
    | :---         |     :---:   |          :---: |
    |VGG16   |  0.5582  |  0.8881 |
--------------
# Akhund and Human 👳🏻‍♂️👨🏻
- A deep learning model using MobileNetV2 Convolutional Neural Network is trained to recognize the human body 👳🏻‍♂️👨🏻
- [Telegram Bot ](https://t.me/Python_and_Ai_bot)
- [wandb](https://wandb.ai/mohamad-nematizadehhh/Akhund%20and%20Human)
- Download the  dataset from link below:
  - [dataset](https://drive.google.com/drive/folders/1awNU2wbo6owr3Pfyy8KIy70T5sdi41VD?usp=sharing)
  - 
- | Dataset |  Loss	| Accuracy 
    | :---         |     :---:   |          :---: |
    |Train  | 0.0093  | 0.9973 |
    |Val     | 0.2555    | 0.9457  |

--------------
# 7.5.CNN Regression
# Age Prediction
- Automatic estimation of human age based on the appearance of human face 👶🏻👵🏻,
- Using ResNet50V2 neural network

- Download the  dataset from link below:
  - [dataset](https://www.kaggle.com/jangedoo/utkface-new)

- [wandb](https://wandb.ai/mohamad-nematizadehhh/Age_Prediction)

- | Dataset |  Loss	     | 
    | :---   |   :---:   | 
    |Train   |  9.2271   | 
    |Val     |  8.7625   |

--------------
# 7.7 OCR(Optical Character Recognition)
# easyOCR
- Extracts the text using easyOCR

| resalt |  	     | 
| :---   |   :---:   | 
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/EasyOcr/input/easyOCR.jpg?raw=true)    | 'Python', 'Web Apps','From Scratch'|
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/EasyOcr/input/2.webp?raw=true)    | ازتو چشد تومی تابه,چشمه چشمه ابر ایثا,روى سينه ى توخو ابه,لوكدوم خليج سبزى|
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/EasyOcr/input/3.jpg?raw=true)    | DMC-4583|
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/EasyOcr/input/Iranian_plate.jpg?raw=true)    | 29٧٢٨٤٣|

# DTRB(Deep Text Recognition Benchmark)
- Vehicle license plate extraction using DTRB
    | Train | Test  | 
    | :---   |   :---:   | 
    | 73.918 | 73.998   | 

- Download the  dataset ,weights  from link below
 [dataset](https://drive.google.com/drive/folders/1hTuK4nj27cyAGaRL3ZqOqbO8gU_9IJCK)
### How to install
```
pip install -r requirements.txt
```

|          Image name        |  predicted_labels   | confidence score |  
| :----------------------:   | :-----------: | :--------: |
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/DTRB/test_imag/00192.jpg?raw=true) | 67e7737 |0.9347|
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/DTRB/test_imag/01656.jpg?raw=true)    | 97i48912|0.9987|
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/DTRB/test_imag/lp3.jpg?raw=true)    | 57c82374|0.9955|
| ![screen shot](https://github.com/MohamadNematizadeh/Deep-Learning/blob/main/7.7.OCR/DTRB/test_imag/lp5.jpg?raw=true)    | 73v96442  |0.8017|

--------------
# Audio Classification  🗣
Audio classification is a fundamental problem in the field of audio processing. It is a challenging problem because there is no clear definition of what is a good representation of audio data. In this project, we will use a custom dataset to classify audio files into 17 classes.Identify sounds in audio clips using Tensorflow and pydub 🗣

# How to Install
Run this command:
```
pip install -r requirements.txt
```
Download [dataset](https://drive.google.com/drive/folders/1F29a-Y00TQ6gsm63rQGqaO9ahHC7KNDV?usp=sharing)

Extract it in the `./raw_audios` directory.

Use `make_dataset.ipynb` notebook to convert the data into a dataset with format that can be used by the model.

## Training

Use `Train.ipynb` notebook to train the model.

[**Bot Telegram**](t.me/Python_and_Ai_bot)
--------------
# Face Recognition 🧑
In this face recognition project using InsightFace
 And pytorch is built 🧑
# How to run
```
pip install -r requirements.txt
```
## Inference:
```
python3 FaceـVerification.py --image1 {yore imag1} --image2 {yore imag2}
```

--------------
# Natural Language Processing (NLP)
##  Emoji Text Classification 📝

A field of AI that enables machines to understand, generate, and interact with human language, revolutionizing content creation and chatbots. 📝



| Feature Vector Dimensions | Train Loss| Train Accuracy | Test Loss| Test Accuracy | Inference Time |
| :---                      |     :---: |       :---:    |    :---: |        :---:  |       :---:    |
| 50d                       | 0.7624    | 0.7500         |   0.7805 | 0.7818        | 0.00148 s      |
| 100d                      | 0.4389    | 0.8864         |   0.5553 | 0.8364        | 0.00188 s      |
| 200d                      | 0.3595    | 0.9015         |   0.5148 | 0.8727        | 0.00105 s      |
| 300d                      | 0.2285    | 0.9621         |   0.4619 | 0.8909        | 0.00102 s      |


# How to install
```
pip install -r requirements.txt
```

Download [weights](https://drive.google.com/drive/folders/1g8_5k6kq7SvyJKh4PitG9Qw72BApYIU2?usp=drive_link) and [glov.6b](https://drive.google.com/drive/folders/1-DSGFjAl2GtOtzJwmZpoK2lbb15NQhsR?usp=drive_link)


ran file main is  `main.ipynb`


