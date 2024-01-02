import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# __Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--image1', type=str , help="Taking the first photo")
parser.add_argument('--image2', type=str , help="Taking a second photo")
opt = parser.parse_args()

# Create FaceAnalysis
app = FaceAnalysis(name="buffalo_s",providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Catch Your imags
imag_1 = cv2.imread(opt.image1)
imag_1 = cv2.cvtColor(imag_1,cv2.COLOR_BGR2RGB)
imag_2 = cv2.imread(opt.image2)
imag_2 = cv2.cvtColor(imag_2,cv2.COLOR_BGR2RGB)

# embeddings
result_1 = app.get(imag_1)
embedding_1 = result_1[0]["embedding"]
result_2 = app.get(imag_2)
embedding_2 = result_2[0]["embedding"]

# Result
if np.sqrt(np.sum((embedding_1 - embedding_2) **2)) < 25:
    print("Same Person ✅ ")
else:
    print("Different Persons 📛")    
