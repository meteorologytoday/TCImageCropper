import sys
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
from ColorTrans import ColorTrans
from DrawSection import drawSection


rad2deg = 180.0 / np.pi
class SamplePreviewer(tk.Frame):
	def __init__(self, parent, oper_img, r_theta_img, center, dtheta, dr):
		super(SamplePreviewer, self).__init__(parent)
		self.parent = parent
		self.oper_img = ImageTk.PhotoImage(oper_img)
		self.r_theta_img = ImageTk.PhotoImage(r_theta_img)
		self.dtheta = dtheta
		self.dr = dr
		self.center = center
		self.sec_r = ( self.oper_img.width() ** 2.0 + self.oper_img.height() ** 2.0 ) ** 0.5
		self.sec_len = int(360.0/dtheta)
		self.box_h = self.r_theta_img.height() / self.sec_len

		self.sec_imgs = [None for _ in range(self.sec_len)]

		self.setupUI()
		self.setupEvent()

		self.light = 0

	def setupUI(self):
		self.tframe = tk.Frame(self)
		self.tframe.pack(side=tk.TOP)

		self.mframe = tk.Frame(self)
		self.mframe.pack(side=tk.TOP)

		self.mlframe = tk.Frame(self.mframe); self.mlframe.pack(side=tk.LEFT)
		self.mrframe = tk.Frame(self.mframe); self.mrframe.pack(side=tk.LEFT)
		
		tk.Button(self.tframe, text="Leave", command=self.parent.destroy).pack(side=tk.LEFT)

		self.status = tk.StringVar()
		tk.Label(self, font="size, 20", textvariable=self.status, height=1).pack(side=tk.LEFT)

		self.drawsec = tk.Label(self.mrframe, image=None)
		self.drawsec.pack(side=tk.TOP)

		self.canvas = tk.Canvas(self.mlframe, width=self.oper_img.width(), height=self.oper_img.height(), relief='ridge', borderwidth=0)
		self.canvas.config(highlightthickness=0, borderwidth=0,closeenough=0)
		
		self.canvas.create_image((0,0), anchor=tk.NW, image=self.oper_img)
		self.canvas.create_line(0,0,0,0, fill='red', tags='section')
		self.canvas.pack(padx=10, pady=10, side=tk.LEFT)


		self.c_rtheta = tk.Canvas(self.mrframe, width=self.r_theta_img.width(), height=self.r_theta_img.height(), relief='ridge', borderwidth=0)
		self.c_rtheta.config(highlightthickness=0, borderwidth=0,closeenough=0)
		self.c_rtheta.create_image((0,0), anchor=tk.NW, image=self.r_theta_img)
		self.c_rtheta.create_rectangle(0,0,0,0, outline="#ffffff", width=1, tags='box')
		self.c_rtheta.pack(side=tk.LEFT)

	def setupEvent(self):
		self.canvas.bind("<Button-1>", self.drawSection)
		self.c_rtheta.bind("<Button-1>", self.drawSection)
		
		self.canvas.bind("<Motion>", self.detectSectionCanvas)
		self.c_rtheta.bind("<Motion>", self.detectSectionRThetaCanvas)

	def drawSection(self, evt):
		global drawSection
		if self.sec_imgs[self.light] is None:
			savename = 'img/sec_%03d.png' % (self.light,)
			drawSection('data/sec_%03d.bin' % (self.light,), savename)
			self.sec_imgs[self.light] = ImageTk.PhotoImage(Image.open(savename))

		self.drawsec.configure(image=self.sec_imgs[self.light])
		

	def detectSectionCanvas(self, evt):
		global rad2deg
		dx = evt.x - self.center[0]
		dy = evt.y - self.center[1]

		theta = np.arccos(- dy / (dx**2.0 + dy**2.0) ** 0.5) * rad2deg
		if dx < 0:  # over 180 degree
			theta = 360.0 - theta

		light = int(np.round(theta / self.dtheta))
		self.status.set("deg: %d" % theta)
		self.lightSection(light)

	def detectSectionRThetaCanvas(self, evt):
		light = int(np.round(evt.y / self.box_h))
		self.status.set("deg: %d" % (int(light * self.dtheta)))
		self.lightSection(light)


	def lightSection(self, light):
		if self.light != light:
			self.light = light
			endx =  self.center[0] + self.sec_r * np.sin(light * self.dtheta / rad2deg)
			endy =  self.center[1] - self.sec_r * np.cos(light * self.dtheta / rad2deg)
			self.canvas.coords('section', self.center[0], self.center[1], endx, endy)
			self.c_rtheta.coords('box', 0, self.box_h * light - 1, self.r_theta_img.width(), self.box_h * (light+1) + 1)


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
		d.tofile("data/total.bin")
		for i in range(d.shape[0]):
			d[i,:].tofile("data/sec_%03d.bin"%(i,))
		print("Image from array...", end=''); sys.stdout.flush()
		self.r_theta_img = Image.fromarray(self.color_trans.temp2colorArray(d))
		self.r_theta_img = self.r_theta_img.resize((self.r_theta_img.width*4, self.r_theta_img.height*4), Image.BOX)
		self.r_theta_img.save("img/r_theta.png")
		self.oper_img.save("img/operation.png")
		
		preview = tk.Toplevel(self)
		SamplePreviewer(preview, self.oper_img, self.r_theta_img, self.center, self.dtheta, self.dr).pack()
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
				posx = int(self.center[0] + r * np.sin(theta))
				posy = int(self.center[1] - r * np.cos(theta))
				if not ( posx >= self.xlim[0] and posx < self.xlim[1] \
					 and posy >= self.ylim[0] and posy < self.ylim[1]):
					continue

				self.cdata[i,j] = self.pixel[posy, posx]
				pix[posy, posx][:] = (255,255,255)
		
		self.oper_img = Image.fromarray(pix)

		

if __name__ == '__main__':

	if len(sys.argv) >= 2:
		imgfile = sys.argv[1]
	else:
		root = tk.Tk()
		root.withdraw()
		imgfile = filedialog.askopenfilename()
		root.destroy()

	print(imgfile)

	view = tk.Tk()
	view.title("Center selector")
	TCWidget(view, imgfile).pack()

	view.mainloop()

