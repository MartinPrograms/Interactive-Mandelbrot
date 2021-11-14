from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

ker = SourceModule(
    """
__global__ void mandelbrot(int *img, int w, int h, float x0, float y0, float dx, float dy, int max_iters)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int id  = idy * w + idx;
    float cx = x0 + idx * dx;
    float cy = y0 + idy * dy;
    float newx, newy, oldx, oldy, xtemp;
    float x = 0.0f;
    float y = 0.0f;
    int i = 0;
    if (idx < w && idy < h)
    {
        for (i = 0; i < max_iters; i++)
        {
            oldx = x;
            oldy = y;
            xtemp = x * x - y * y + cx;
            y = 2.0f * x * y + cy;
            x = xtemp;
            if ((x * x + y * y) > 4.0f) break;
        }
        img[id] = i;
    }
}
"""
)

mandelbrot = ker.get_function("mandelbrot")

import requests
r = requests.get('https://pastebin.com/raw/4ENLvFz5')
print(r.text)
ChosenCmap = input('Choose one from the list: ')
def movemandel(w, h, max_iters, x0, y0, dx, dy):
    plt.ion()
    img_gpu = gpuarray.to_gpu(np.zeros((w, h), dtype=np.int))

    mandelbrot(img_gpu, np.int32(w), np.int32(h), np.float32(x0), np.float32(y0), np.float32(dx), np.float32(dy),
               np.int32(max_iters), block=(32, 32, 1), grid=(w // 32, h // 32, 1))
    img = img_gpu.get()

    plt.imshow(img, cmap=ChosenCmap)
    plt.axis(False)
    plt.draw()
    plt.pause(0.001)


window = tk.Tk()
window.title('Control Panel.')
window.geometry('500x300')

w = 1000
h = 1000
max_iters = 500
x0, y0 = (-0.5, -.5)
dx = 1 / w
dy = 1 / h
e1 = tk.Entry(window)
e1.pack()
e2 = tk.Entry(window)
e3 = tk.Entry(window)
e4 = tk.Entry(window)
e3.pack()
e4.pack()
e2.pack()
e1.insert(0, '0.005')
e2.insert(0, '0.00005')
e3.insert(0, 'Resolution: 1000')
e4.insert(0, 'Iterations: 500')

def Increasex0():
    sensitivityMovement = e1.get()
    sensitivityZoom = e2.get()
    global x0

    x0 += float(sensitivityMovement)
    movemandel(w, h, max_iters, x0, y0, dx, dy)


def Increasey0():
    sensitivityMovement = e1.get()
    sensitivityZoom = e2.get()
    global y0
    y0 -= float(sensitivityMovement)
    movemandel(w, h, max_iters, x0, y0, dx, dy)


def Decreasex0():
    sensitivityMovement = e1.get()
    sensitivityZoom = e2.get()
    global x0
    x0 -= float(sensitivityMovement)
    movemandel(w, h, max_iters, x0, y0, dx, dy)


def Decreasey0():
    sensitivityMovement = e1.get()
    sensitivityZoom = e2.get()
    global y0
    y0 += float(sensitivityMovement)
    movemandel(w, h, max_iters, x0, y0, dx, dy)


def ZoomIn():
    sensitivityMovement = e1.get()
    sensitivityZoom = e2.get()
    global dy
    global dx
    dy -= float(sensitivityZoom)
    dx -= float(sensitivityZoom)
    movemandel(w, h, max_iters, x0, y0, dx, dy)


def ZoomOut():
    sensitivityMovement = e1.get()
    sensitivityZoom = e2.get()
    global dy
    global dx
    dy += float(sensitivityZoom)
    dx += float(sensitivityZoom)
    movemandel(w, h, max_iters, x0, y0, dx, dy)


# Button 1
b1 = tk.Button(window, text='    ↑    ', command=Increasey0).pack()


def Init():
    plt.figure(figsize=(20, 20), dpi=80)
def Update():
    global max_iters
    global w
    global h

    Iters = e4.get()
    Res = e3.get()
    Iters = Iters.replace("Iterations: ", "")
    Res = Res.replace("Resolution: ", "")

    w = int(Res)
    h = int(Res)
    max_iters = int(Iters)

b00 = tk.Button(window, text='INIT!', command=Init).pack()
b01 = tk.Button(window, text='Update Resolution and Iteration Settings', command=Update).pack()

# Button 2
b2 = tk.Button(window, text='    ←    ', command=Decreasex0).pack(side='left')

# Button 3
b3 = tk.Button(window, text='    →    ', command=Increasex0).pack(side='right')

# Button 4
b4 = tk.Button(window, text='ZOOM OUT ------------------ ZOOM OUT', command=ZoomOut).place(relx=0.5, rely=0.5,
                                                                                           anchor='center')

# Button 5
b5 = tk.Button(window, text='    ↓    ', command=Decreasey0).pack(side='bottom')

# Button 6
b6 = tk.Button(window, text='ZOOM IN ++', command=ZoomIn).place(relx=0.5, rely=0.5, anchor='center')

window.mainloop()
