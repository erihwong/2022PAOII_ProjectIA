from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import pickle
from googlesearch import search
from colores import *

# import tensorflow.keras as keras
import tensorflow as tf

keras = tf.keras
import keras.utils as image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from rembg import remove
import cv2
from sklearn.cluster import KMeans
from collections import Counter

root = Tk()
root.title("Buscador de ropa a menor precio - Grupo 5")


def obtener_nombre_clase(index):
    test_dir = r'..\ModeloClasificadorRopa\Test'
    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=3,
                                                      class_mode='categorical')
    for class_string, class_index in test_generator.class_indices.items():
        if class_index == index:
            return class_string


def create_model():
    vgg16 = keras.applications.vgg16
    conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
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
    modelo_CR.load_weights(r'.\modelo_final_CR_.h5')

    img = image.load_img(imagen_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # se la pasa por el modelo para obtener una prediccion
    imagen_prediccion = modelo_CR.predict(x)
    indice_clase_prediccion = np.argmax(imagen_prediccion, axis=1)

    # label a usarse en busqueda
    label_prediccion = obtener_nombre_clase(indice_clase_prediccion)

    return label_prediccion


def removerBackground(imagen_path):
    input_path = imagen_path

    if 'jpg' in input_path:
        x = input_path.split(".jpg")

        output_path = x[0] + "SBG.png"

    if 'png' in input_path:
        x = input_path.split(".png")

        output_path = x[0] + "SBG.png"

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
    resized_imagen = imagen.reshape(imagen.shape[0] * imagen.shape[1], 3)

    # clasificacion por clusters de los colores
    clf = KMeans(n_clusters=num_colores)
    labels = clf.fit_predict(resized_imagen)

    num_labels = Counter(labels)

    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i] for i in num_labels.keys()]

    # arreglo de resultado
    rgb_colors = [getRGB(ordered_colors[i]) for i in num_labels.keys()]
    return rgb_colors


def filtrar_colores(imgPath) -> str:
    ruta_imagen_sinBG = removerBackground(imgPath)
    lista_colores_rgb = getColoresImg(ruta_imagen_sinBG)
    lista_colores_nombres = []
    for color in lista_colores_rgb:
        nombre_color = findclosest(color)
        lista_colores_nombres.append(nombre_color)
    nombre_color = max(set(lista_colores_nombres), key=lista_colores_nombres.count)
    return nombre_color


def getImg():
    global imgNew
    root.filename = filedialog.askopenfilename(initialdir="./images", title="Select a file",
                                               filetypes=(("png files", "*.png"), ("jpg files", "*.jpg")))
    imgNew = Image.open(root.filename)
    imgNew = imgNew.resize((300, 300), Image.LANCZOS)
    imgNew = ImageTk.PhotoImage(imgNew)
    imgLabel["image"] = imgNew
    print(root.filename)

    root.tipo_prenda = clasificarPrenda(root.filename)
    root.colores_rgb = filtrar_colores(root.filename)

    #print("color: "+ root.tipo_prenda)
    #print(root.colores_rgb)


def searchGoogle():
    query_example_tipo = root.tipo_prenda
    query_example_color = root.colores_rgb
    price = inputField_price.get()

    query = query_example_tipo + " " + query_example_color + " a " + price + " dolares o menos"
    resultados = []

    for j in search(query, lang="es", country="countryEC", tld="co.in", num=5, stop=5, pause=2):
        resultados.append(j)

    myLabel2 = Label(secondFrame, text="Resultados de busqueda")
    myLabel2.grid(row=4, column=0, columnspan=2)
    textBox = Text(secondFrame, height=15, width=60)
    textBox.grid(row=5, column=0, columnspan=2)
    textBox.insert(END, '-' + resultados[0] + '\n')
    textBox.insert(END, '-' + resultados[1] + '\n')
    textBox.insert(END, '-' + resultados[2] + '\n')
    textBox.insert(END, '-' + resultados[3] + '\n')
    textBox.insert(END, '-' + resultados[4] + '\n')
    textBox.config(state="disable")

    recommendationQuery = searchRecommendations(query)
    resultados2 = []

    for j in search(recommendationQuery, lang="es", country="countryEC", tld="co.in", num=5, stop=5, pause=2):
        resultados2.append(j)

    myLabel3 = Label(secondFrame, text="Recomendacion de productos similares")
    myLabel3.grid(row=6, column=0, columnspan=2)
    textBox2 = Text(secondFrame, height=15, width=60)
    textBox2.grid(row=7, column=0, columnspan=2)
    textBox2.insert(END, '-' + resultados2[0] + '\n')
    textBox2.insert(END, '-' + resultados2[1] + '\n')
    textBox2.insert(END, '-' + resultados2[2] + '\n')
    textBox2.insert(END, '-' + resultados2[3] + '\n')
    textBox2.insert(END, '-' + resultados2[4] + '\n')
    textBox2.config(state="disable")


def get_recommendations(product, model, vector):
    #print("Cluster ID:")
    Y = vector.transform([product])
    prediction = model.predict(Y)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vector.get_feature_names_out()
    
    recommendationQuery = ""
    for ind in order_centroids[prediction, :10]:
        recommendationQuery = recommendationQuery + str(terms[ind])
    recommendationQuery = recommendationQuery.replace("'", "")
    recommendationQuery = recommendationQuery.replace("[", "")
    recommendationQuery = recommendationQuery.replace("]", "")
    return recommendationQuery
        

def searchRecommendations(query):
    loaded_model = pickle.load(open("modelo_recomendaciones.sav", 'rb'))
    loaded_vectorizer = pickle.load(open("vector_recomendaciones.sav", 'rb'))
    return get_recommendations(query, loaded_model, loaded_vectorizer)


# Create A Main Frame
mainFrame = Frame(root)
mainFrame.pack(fill=BOTH, expand=1)

# Create A Canvas
myCanvas = Canvas(mainFrame)
myCanvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add A Scrollbar To The Canvas
myScrollbar = ttk.Scrollbar(mainFrame, orient=VERTICAL, command=myCanvas.yview)
myScrollbar.pack(side=RIGHT, fill=Y)

# Configure canvas
myCanvas.configure(yscrollcommand=myScrollbar.set)
myCanvas.bind('<Configure>', lambda e: myCanvas.configure(scrollregion=myCanvas.bbox("all")))

# Second Frame
secondFrame = Frame(myCanvas)

myCanvas.create_window((0, 0), window=secondFrame, anchor="nw")

# Creating widgets
btnLoadImg = Button(secondFrame, text="SELECCIONAR IMAGEN", command=getImg, padx=60)

imgDefault = Image.open("default.jpg")
imgDefault = imgDefault.resize((300, 300), Image.LANCZOS)
imgDisplay = ImageTk.PhotoImage(imgDefault)
imgLabel = Label(secondFrame, image=imgDisplay)

myLabel1 = Label(secondFrame, text="Ingrese el precio de referencia:")
inputField_price = Entry(secondFrame)

btnSearch = Button(secondFrame, text="REALIZAR BUSQUEDA", command=searchGoogle)

#myLabel3 = Label(root, text="Recomendaciones de productos relacionados")

# Shoving widgets onto the screen
btnLoadImg.grid(row=0, column=0, columnspan=2)
imgLabel.grid(row=1, column=0, columnspan=2)

myLabel1.grid(row=2, column=0, padx=1, pady=1)
inputField_price.grid(row=2, column=1)

btnSearch.grid(row=3, column=0, columnspan=2)

#myLabel3.grid(row=4, column=1)

root.mainloop()
