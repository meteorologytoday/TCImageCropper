import numpy as np

class ColorTrans:
	"""
	Default to work with 2D images.
	"""
	def __init__(self):
		"""
		The temperature navy800.txt is assumed to be equally spaced and increasing.
		"""
		self.ctab = np.loadtxt("navy800.txt", unpack=False, usecols=(1,2,3,4,0))
		self.temp_interval = self.ctab[1][3] - self.ctab[0][3]

	def isData(self, color):
		diff = abs(color - np.roll(color,1))

		if (diff < 20).all() or (color <= 5).all() or (color >= 128).all():
			return False

		return True

	def temp2color(self, temp):
		idx = int((temp - self.ctab[0][3]) / self.temp_interval)
		if idx < 0 or idx > len(self.ctab):
			raise Exception("Temperature out of range: %f" % (temp,))

		return self.ctab[idx][0:3]

	def color2temp(self, color):
		"""
		This method returns the temperature represented by [color]
		which is an (R, G, B) tuple/list. Each element rages from
		0 to 255. The distance function	used is	eulidean metric.
		"""
		i_min = 0
		d_min = 255 ** 3
		c_min = np.zeros(3)
		for i, c in enumerate(self.ctab):
			d = ((c[:3] - color) ** 2.0 ).sum()
			if d < d_min:
				i_min, d_min = i, d
	
		return self.ctab[i_min][3]
	
	def color2tempArray(self, carr, missing=-999.0):
		result = np.zeros((carr.shape[0], carr.shape[1]))
		for idx in np.ndindex(carr.shape[:-1]):
			c = carr[idx]
	
			if self.isData(c):
				result[idx] = self.color2temp(c)
			else:
				result[idx] = missing

		return result
	
	def temp2colorArray(self, tarr, missing=-999.0):
		sh = tarr.shape
		result = np.zeros((sh[0], sh[1], 3), dtype=np.uint8)
		
		for idx, temp in np.ndenumerate(tarr):
			if temp == missing:
				result[idx] *= 0
			else:
				result[idx] = self.temp2color(temp)

		return result


