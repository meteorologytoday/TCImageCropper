import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pplt
import sys, os

if not os.path.isdir('./img'):
	os.mkdir('./img')


def drawSection(filename, savename):
	data = np.fromfile(filename, dtype='<f4')
	for i, val in enumerate(data):
		if val == -999.0:
			data[i] = np.nan

	# calculate figure size
	graph_size = [8.0, 3.0]
	space = {
		'tspace': 0.4, 
		'bspace': 0.4,
		'lspace': 0.8,
		'rspace': 0.3
	}
	figsize = [graph_size[0] + space['lspace'] + space['rspace'],
    	       graph_size[1] + space['tspace'] + space['bspace']]

	fig = pplt.figure(figsize=figsize)
	figdpi=72

	# create main axes
	rect = (
		space['lspace'] / figsize[0],
		space['bspace'] / figsize[1],
		graph_size[0]   / figsize[0],
		graph_size[1]   / figsize[1]
	)

	ax = fig.add_axes(rect, autoscale_on=True)
	#	ax.set_xlim(t_range)
	#	ax.set_xticks([])
	ax.set_ylim([150, 290])

	x = np.arange(0, len(data))
	ax.set_title(filename)
	ax.plot(x, data)
	fig.savefig(savename, dpi=figdpi)
	
