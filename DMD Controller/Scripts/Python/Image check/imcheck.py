#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Jan. 28th, 2019
Last modification: Mar. 13th, 2019
MORGADO, Manuel (M2-MCN , QMAT fellow)
U. Strasbourg // ISIS // IPCMS
Supervisor. Prf. Dr. S. Whitlock

IMAGE CHECK, FOR DMD FORMAT

"""
####################################################################################################

"""
TO DO List:

    - THIS IS JUST AN AUXILIAR SCRIPT
"""

####################################################################################################
####| HERE IMPORTING LYBS |####
try:
	import numpy as np 
	import matplotlib.pyplot as 

	from PIL import Image 
	from PIL.ExifTags import TAGS, GPSTAGS

	import numpy as np 
	import matplotlib.pyplot as 

	from skimage import data, color
	from skimage.transform import rescale, resize, 

	print('Succesfully import of packages.')
except:
	print('Error importing packages.')

#importing images to be compare	
pic1 = 'image1.png'
pic2 = 'image2.png'

#saving images
image1 = Image.open(pic1)
image2 = Image.open(pic2)

#printing data of images
# print(image1)
# print(image2)

#extracting and substituying palettes of both images
pal1=image1.getpalette()
# image2.putpalette(pal1)
image2.convert("P", palette=pal1, colors=1)
pal2=image2.getpalette()

#function to cut images for fit in DMD
def crop(imagen, fshape):
	x,y = np.shape(imagen)
	difX = int(abs(fshape[0] - x)/2)
	difY = int(abs(fshape[1] - y)/2)
	box = (difX , difY, difX+fshape[0], difY+fshape[1])
	imagen.crop(box)

#plot of images
plt.figure(1)
plt.imshow(image2)
plt.show()