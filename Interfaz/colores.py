import math

colors = [
    ("negro", (0, 0, 0)),
    ("gris", (128, 128, 128)),
    ("blanco", (255, 255, 255)),
    ("marron", (128, 0, 0)),
    ("rojo", (255, 0, 0)),
    ("morado", (128, 0, 128)),
    ("verde", (0, 128, 0)),
    ("amarillo", (255, 255, 0)),
    ("azul", (0, 0, 255)),
]

def distance(a,b):
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    dz = a[2]-b[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def findclosest(pixel):
    mn = 999999
    for name,rgb in colors:
        d = distance(pixel, rgb)
        if d < mn:
            mn = d
            color = name
    return color