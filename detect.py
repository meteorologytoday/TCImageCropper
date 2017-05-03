import sys
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk

box_w=5
box_s=int(box_w/2)
def coord(evt):
	global msg
	msg.set("%d, %d" % (evt.x, evt.y))


def click(evt):
	global canvas
	canvas.coords('box', evt.x - box_s, evt.y - box_s, evt.x+box_s, evt.y+box_s)
	canvas.update()

dragging = False
select_box = [0,0,0,0]
missing = -999.0

ctab = np.loadtxt("navy800.txt", unpack=False, usecols=(1,2,3,4,0))

def select_beg(evt):
	global dragging, select_box
	dragging = True
	select_box[0], select_box[1] = evt.x, evt.y
	
	
def select_drag(evt):
	global canvas, select_box
	if dragging:
		select_box[2], select_box[3] = evt.x, evt.y
		canvas.coords('select', *select_box)
	

def select_end(evt):
	global dragging
	dragging = False


def isData(color):

	diff = abs(color - np.roll(color,1))

	if (diff < 20).all() or (color <= 5).all() or (color >= 128).all():
		return False

	return True

print(ctab)
def color2temp(color):
	"""
	This method returns the temperature represented by [color]
	which is an (R, G, B) tuple/list. Each element rages from
	0 to 255. The distance function	used is	eulidean metric.
	"""
	global ctab

	i_min = 0
	d_min = 255 ** 3
	c_min = np.zeros(3)
	for i, c in enumerate(ctab):
		d = ((c[:3] - color) ** 2.0 ).sum()
		if d < d_min:
			i_min, d_min = i, d

	return ctab[i_min][3]

def color2tempArray(carr, missing=-999.0):
	result = np.zeros(carr.shape)
	for idx in np.ndindex(carr.shape[:2]):
		print(idx)
		c = carr[idx]

		if isData(c):
			result[idx] = color2temp(c)
		else:
			result[idx] = missing

	return result

def temp2colorArray(tarr, missing=-999.0):
	result = np.zeros(tarr.shape)
#	for idx, temp in np.ndenumerate(tarr):
#		if temp == missing:
#			result[idx] = color2temp(c)
#		else:
#			result[idx] = missing



def color2tempAction():
	global pixel
#	newimg = Image.fromarray(color2tempArray(pixel))
	color2tempArray(pixel).tofile("test.bin")

#	newimg.save("test.png")


imgfile = sys.argv[1]

img = Image.open(imgfile)
pixel = np.array(img)


view = tk.Tk()
view.title("Center selector")

tkimg = ImageTk.PhotoImage(img)

canvas = tk.Canvas(view, width=tkimg.width(), height=tkimg.height(), relief='ridge', borderwidth=0)
canvas.config(highlightthickness=0, borderwidth=0,closeenough=0)
print(canvas.config())

canvas.create_image((0,0), anchor=tk.NW, image=tkimg)
canvas.create_oval(0,0, box_w, box_w, fill="#ff0000", tags='box')
canvas.create_rectangle(0,0, 0, 0, outline="#000000", width=5, tags='select')
canvas.pack(padx=10, pady=10)

msg = tk.StringVar()
tk.Label(view, font="size, 20", textvariable=msg, height=1).pack()
canvas.bind("<Motion>", coord)
canvas.bind("<Button-1>", click)

canvas.bind("<ButtonPress-3>", select_beg)
canvas.bind("<ButtonRelease-3>", select_end)
canvas.bind("<Motion>", select_drag, add="+")


tk.Button(view, text="color2temp", command=color2tempAction).pack()


view.mainloop()

"""






for i in range(pixel.shape[0]):
	for j in range(pixel.shape[1]):
		r, g, b = pixel[i,j]
		print("%d,%d : (%d, %d, %d) " % (i,j,r,g,b))
"""
