import sys
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from ColorTrans import ColorTrans

class SamplePreviewer(tk.Frame):
	def __init__(self, parent, oper_img, r_theta_img, center, dtheta, dr):
		super(SamplePreviewer, self).__init__(parent)
		self.parent = parent
		self.oper_img = ImageTk.PhotoImage(oper_img)
		self.r_theta_img = ImageTk.PhotoImage(r_theta_img)
		self.dtheta = dtheta
		self.dr = dr
		self.center = center

		self.setupUI()
		self.setupEvent()

	def setupUI(self):
		self.tframe = tk.Frame(self)
		self.tframe.pack(side=tk.TOP)

		self.mframe = tk.Frame(self)
		self.mframe.pack(side=tk.TOP)
		
		tk.Button(self.tframe, text="Leave", command=self.parent.destroy).pack(side=tk.LEFT)
		
		self.canvas = tk.Canvas(self.mframe, width=self.oper_img.width(), height=self.oper_img.height(), relief='ridge', borderwidth=0)
		self.canvas.config(highlightthickness=0, borderwidth=0,closeenough=0)
		
		self.canvas.create_image((0,0), anchor=tk.NW, image=self.oper_img)
		self.canvas.pack(padx=10, pady=10, side=tk.LEFT)

		self.r_theta_img_label = tk.Label(self.mframe, image=self.r_theta_img)
		self.r_theta_img_label.pack(side=tk.LEFT)

	
	def setupEvent(self):
		pass


class TCWidget(tk.Frame):
	def __init__(self, parent, imgfile):
		super(TCWidget, self).__init__(parent)
		self.parent = parent
		
		self.box_w=5
		self.box_s=int(self.box_w/2)
		self.dragging = False
		self.select_box = np.array([0,0,0,0])
		self.center = np.array([0,0])
		self.color_trans = ColorTrans()
		self.dr=2.0
		self.dtheta=5.0
		
		self.imgfile = imgfile
		self.img = Image.open(self.imgfile)
		self.pixel = np.array(self.img)
		self.tkimg = ImageTk.PhotoImage(self.img)
		self.imgsize = np.array([self.tkimg.width(), self.tkimg.height()])
		self.oper_img = None
		self.r_theta_img = None

		self.setupUI()
		self.setupEvent()

	def setupEvent(self):
		self.canvas.bind("<Motion>", self.coord)
		self.canvas.bind("<Button-1>", self.recenter)
		
		self.canvas.bind("<ButtonPress-3>", self.select_beg)
		self.canvas.bind("<ButtonRelease-3>", self.select_end)
		self.canvas.bind("<Motion>", self.select_drag, add="+")

	def setupUI(self):
	
		self.tframe = tk.Frame(self)
		self.tframe.pack(side=tk.TOP)

		tk.Button(self.tframe, text="color2temp", command=self.color2tempAction).pack(side=tk.LEFT)
		tk.Button(self.tframe, text="Generate", command=None).pack(side=tk.LEFT)
		tk.Button(self.tframe, text="Leave", command=self.parent.destroy).pack(side=tk.LEFT)

		self.canvas = tk.Canvas(self, width=self.tkimg.width(), height=self.tkimg.height(), relief='ridge', borderwidth=0)
		self.canvas.config(highlightthickness=0, borderwidth=0,closeenough=0)
		
		self.canvas.create_image((0,0), anchor=tk.NW, image=self.tkimg)
		self.canvas.create_oval(0,0, self.box_w, self.box_w, fill="#ff0000", tags='box')
		self.canvas.create_rectangle(0,0, 0, 0, outline="#000000", width=5, tags='select')
		self.canvas.pack(padx=10, pady=10, side=tk.TOP)
		
		self.status = tk.StringVar()
		tk.Label(self, font="size, 20", textvariable=self.status, height=1).pack(side=tk.TOP)

	def updateCropSize(self):
		self.xlim = sorted([self.select_box[0], self.select_box[2]])
		self.ylim = sorted([self.select_box[1], self.select_box[3]])

	def select_beg(self, evt):
		self.dragging = True
		self.select_box[0], self.select_box[1] = evt.x, evt.y
		
	def select_drag(self, evt):
		if self.dragging:
			self.select_box[2], self.select_box[3] = evt.x, evt.y
			self.canvas.coords('select', *self.select_box)
		
	def select_end(self, evt):
		self.dragging = False

	def coord(self, evt):
		self.status.set("%d, %d" % (evt.x, evt.y))

	def recenter(self, evt):
		self.center[0] = evt.x
		self.center[1] = evt.y
		self.canvas.coords('box', evt.x - self.box_s, evt.y - self.box_s, evt.x + self.box_s, evt.y + self.box_s)
		self.canvas.update()

	def color2tempAction(self):
		print("Extract Data"); sys.stdout.flush()
		self.extractData()
		d = self.color_trans.color2tempArray(self.cdata).astype(np.float32)
		d.tofile("test.bin")
		for i in range(d.shape[0]):
			d[i,:].tofile("test%03d.bin"%(i,))
		print("Image from array..."); sys.stdout.flush()
		self.r_theta_img = Image.fromarray(self.color_trans.temp2colorArray(d))
		self.r_theta_img = self.r_theta_img.resize((self.r_theta_img.width*4, self.r_theta_img.height*4), Image.BOX)
		self.r_theta_img.save("r_theta.png")
		self.oper_img.save("operation.png")
		
		preview = tk.Toplevel(self)
		SamplePreviewer(preview, self.oper_img, self.r_theta_img, self.center, self.dtheta, self.dr).pack()
		#self.oper_img.show()
		#self.r_theta_img.show()
		print("Done")

	def extractData(self):
		self.updateCropSize()
		pix = self.pixel.copy()
		cropsize = np.array([self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]])
		self.cdata = np.zeros((int(360/self.dtheta), int(np.sqrt(np.sum( cropsize** 2.0))/self.dr), 3))
		#print(self.cdata.shape)
		print("center: %d, %d" % (self.center[0], self.center[1]))
		print("box: %d, %d, %d, %d" % (self.select_box[0],self.select_box[1],self.select_box[2],self.select_box[3],))


		for i in range(self.cdata.shape[0]):
			theta = (self.dtheta * i) * np.pi / 180.0
			for j in range(self.cdata.shape[1]):
				r = self.dr * j
				posx = int(self.center[0] + r * np.cos(theta))
				posy = int(self.center[1] - r * np.sin(theta))
				if not ( posx >= self.xlim[0] and posx < self.xlim[1] \
					 and posy >= self.ylim[0] and posy < self.ylim[1]):
					continue

				self.cdata[i,j] = self.pixel[posy, posx]
				pix[posy, posx][:] = (255,255,255)
		
		self.oper_img = Image.fromarray(pix)

		

if __name__ == '__main__':

	imgfile = sys.argv[1]

	view = tk.Tk()
	view.title("Center selector")
	TCWidget(view, imgfile).pack()

	view.mainloop()

