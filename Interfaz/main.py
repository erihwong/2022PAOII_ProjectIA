from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from googlesearch import search

root = Tk()
root.title("Buscador de ropa a menor precio - Grupo 5")


def getImg():
    global imgNew
    root.filename = filedialog.askopenfilename(initialdir="./images", title="Select a file",
                                               filetypes=(("png files", "*.png"), ("jpg files", "*.jpg")))
    imgNew = Image.open(root.filename)
    imgNew = imgNew.resize((300, 300), Image.LANCZOS)
    imgNew = ImageTk.PhotoImage(imgNew)
    imgLabel["image"] = imgNew
    print(root.filename)


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
