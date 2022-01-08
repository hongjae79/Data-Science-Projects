# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:50:44 2021

@author: hongj
"""
import face_recognition as fr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from emotion_detect import create_model

model = create_model()
model.load_weights('emotion_weights')

def get_faces_with_prob(image):
    img = fr.load_image_file(image) #<- enter with quotation mark.  jpg file
    f = fr.face_locations(img, model = 'hog')
    im = Image.open(image)
    

    faces = []
    faces_color = []

    for i in range(len(f)):
        im_crop = im.crop((f[i][3], f[i][0], f[i][1], f[i][2]))
        X = im_crop
        X = X.resize((48,48))
        Y = X
        Y = np.array(Y)
        X = X.convert('L')
        X = np.array(X)
        faces.append(X)
        faces_color.append(Y)

    z = np.asarray(faces)
    test_prob = model.predict(z)

    return faces_color, test_prob

def display_faces_with_prob(image):
    z = check_if_face_detected(image)
    if z == "no face":
        return print("No face detected")
    else:
        faces_color, test_prob = get_faces_with_prob(image)
        for i in range(len(test_prob)):
            plt.imshow(faces_color[i])#, cmap = 'gray')
            plt.show()    
            for j in range(len(label_map)):        
                print(label_map[j] + ": %4.2f %%" % (test_prob[i][j]*100))
                
def check_if_face_detected(image):   
    img = fr.load_image_file(image)
    face_locations = fr.face_locations(img, model = 'hog')
    if len(face_locations) == 0:
        return "no face"
    else:
        return "yes face"
    
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  

def emotion_indicator(number):
    return label_map[number]

def get_pred_in_words(image):
    faces_color, test_prob = get_faces_with_prob(image)
    test_pred = np.argmax(test_prob, axis=1)
     
    test_pred_in_words = []

    for pred in test_pred:
        test_pred_in_words.append(emotion_indicator(pred))
        
    return test_pred_in_words
