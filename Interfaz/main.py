from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from googlesearch import search

import os
import tensorflow.keras as keras
import tensorflow as tf
import keras.utils as image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from rembg import remove
import cv2
from sklearn.cluster import KMeans
from collections import Counter

root = Tk()
root.title("Buscador de ropa a menor precio - Grupo 5")

def obtener_nombre_clase(index):
    test_dir= r'..\ModeloClasificadorRopa\Test'
    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=3, class_mode='categorical')
    for class_string, class_index in test_generator.class_indices.items():
      if class_index == index:
         return class_string

def create_model():

    vgg16 = keras.applications.vgg16
    conv_model = vgg16.VGG16(weights='imagenet', include_top=False)
    x = keras.layers.Flatten()(conv_model.output)

    
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dense(100, activation='relu')(x)

    
    predictions = keras.layers.Dense(17, activation='softmax')(x)

    
    full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)

    full_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                  metrics=['acc'])

    return full_model

def clasificarPrenda(imagen_path):

    modelo_CR = create_model()
    modelo_CR.load_weights(r'..\modelo_final_CR.h5')

    img = image.load_img(imagen_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    #se la pasa por el modelo para obtener una prediccion
    imagen_prediccion=modelo_CR.predict(x)
    indice_clase_prediccion = np.argmax(imagen_prediccion, axis = 1)


    #label a usarse en busqueda
    label_prediccion = obtener_nombre_clase(indice_clase_prediccion)

    return label_prediccion

def removerBackground(imagen_path):

    input_path = imagen_path

    if 'jpg' in input_path:
        x = input_path.split(".jpg")
        
        output_path = x[0]+"SBG.jpg"
    
    if 'png' in input_path:
        x = input_path.split(".png")
        
        output_path = x[0]+"SBG.png"

    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)

    return output_path

def get_imagen(imagen_path):
    imagen = cv2.imread(imagen_path)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen

def getRGB(color):
    return (int(color[0]), int(color[1]), int(color[2]))

def getColoresImg(imagen_pathSB):
    imagen = get_imagen(imagen_pathSB)
    
    num_colores = 5
    resized_imagen = imagen.reshape(imagen.shape[0]*imagen.shape[1], 3)

    #clasificacion por clusters de los colores
    clf = KMeans(n_clusters = num_colores)
    labels = clf.fit_predict(resized_imagen)

    num_labels = Counter(labels)

    center_colors = clf.cluster_centers_


    ordered_colors = [center_colors[i] for i in num_labels.keys()]

    #arreglo de resultado
    rgb_colors = [getRGB(ordered_colors[i]) for i in num_labels.keys()]
    return rgb_colors

def getImg():
    global imgNew
    root.filename = filedialog.askopenfilename(initialdir="./images", title="Select a file",
                                               filetypes=(("png files", "*.png"), ("jpg files", "*.jpg")))
    imgNew = Image.open(root.filename)
    imgNew = imgNew.resize((300, 300), Image.LANCZOS)
    imgNew = ImageTk.PhotoImage(imgNew)
    imgLabel["image"] = imgNew
    print(root.filename)
    
    tipo_prenda = clasificarPrenda(root.filename)

    ruta_imagen_sinBG = removerBackground(root.filename)

    colores_rgb = getColoresImg(ruta_imagen_sinBG)


    


def searchGoogle():
    price = inputField_price.get()
    query = "camisa"
    for j in search(query, tld="co.in", num=10, stop=10, pause=2):
        print(j)
    #print("el precio" + price)
    #print(root.filename)


# Creating widgets
btnLoadImg = Button(root, text="seleccionar imagen", command=getImg, padx=60)

imgDefault = Image.open("default.jpg")
imgDefault = imgDefault.resize((300, 300), Image.LANCZOS)
imgDisplay = ImageTk.PhotoImage(imgDefault)
imgLabel = Label(root, image=imgDisplay)

myLabel1 = Label(root, text="Ingrese el precio de referencia:")
inputField_price = Entry(root)

btnSearch = Button(root, text="realizar busqueda", command=searchGoogle)

# Shoving widgets onto the screen
btnLoadImg.grid(row=0, column=0, columnspan=2)
imgLabel.grid(row=1, column=0, columnspan=2)

myLabel1.grid(row=2, column=0, padx=1, pady=1)
inputField_price.grid(row=2, column=1)

btnSearch.grid(row=3, column=0, columnspan=2)

root.mainloop()
