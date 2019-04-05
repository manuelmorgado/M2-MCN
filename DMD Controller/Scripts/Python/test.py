
import numpy as np 
from matplotlib import pyplot as plt

import scipy.misc
from scipy import ndimage

import skimage as sk

from PIL import Image 
# from PIL.ExifTags import TAGS, GPSTAGS

resX = 1024;
resY = 768;

x = np.arange(-resX/2,resX/2);   
y = np.arange(-resY/2,resY/2);

origin = [int(0), int(0)];

mesh = np.meshgrid(x,y);

#function for save image
def saveIm(imagen, namef='image.png' ):
	scipy.misc.toimage(imagen, cmin=0, cmax=1).save(namef);

#function that exports the image to binary
def Im2bin(pat, namef='image.png'):
	npat = pat.astype('bool');
	return npat

#function that prepares the image obtained to a format readable by the DMD
def dmd_prep(refim, finalim, fname, angval=0, mode=False):

	pic1 = refim 	#reference image for extract the palette
	pic2 = finalim	#image to be save

	image1 = Image.open(pic1)
	image2 = Image.open(pic2)

	pal1=image1.getpalette() #getting palette
	image2.putpalette(pal1)	 #assigning the reference palette in the final image's palette

	#check if there is any rotation in order to achieve a crop
	if angval != 0:
		image2 = imcrop(image2, (resX,resY))

	image2.save(fname + ".PNG", dpi=(72,72), bits=1)

diskp = np.zeros((resY, resX), dtype = int);
profile=[8,10,20,35,60,90,120,160,200,250,300]#,400, 600, 800, 1000, 1200]
gap=2
ref_im = 'refim.png'

#function of the circular site generator
def Osite(mesh, i, j, pat, val,apRad):
        pat[np.logical_and( np.abs( (mesh[0]-j)**2 ) < apRad/2 - np.abs((mesh[1]-i)**2) , np.abs( (mesh[1]-i)**2 ) < apRad/2 - np.abs( (mesh[0]-j)**2 ))] = val;
        # NOTE: circles are no perfect when rotates because rotate command does not allows define exact coordinate

def disk(mesh, center, pat, profile, gap):
	#copy of the original pattern to be modify
	npat = np.copy(pat);
	
	rad1=[center[0]]
	rad2=list();

	for j in range(len(profile)):		

		r2j=float(rad1[int(j)]) + float(profile[int(j)]);
		rad2.append(r2j)

		r1jj=float(rad2[int(j)]) + float(gap)*0.3;
		rad1.append(r1jj)

		value=0

		for i in rad1[::-1]:
			value = (value+1)%2
			Rad = i*2;
			circ = Osite(mesh, 0, 0, pat, value,Rad)


	# for r1,r2 in zip(rad1, rad2):
		# np.logical_and( r1 < np.abs( mesh[1] )**2  + np.abs( mesh[0] )**2 , np.abs( mesh[0] )**2 + np.abs( mesh[1] )**2 < r2  )=True
		# np.logical_and( r1 < np.abs( mesh[0] )**2  + np.abs( mesh[1] )**2 , np.abs( mesh[0] )**2 + np.abs( mesh[1] )**2 < r2  )=True
		# print(np.shape(npat))
		# print(mesh[0], mesh[1])
		# npat[ np.logical_and( mesh[1]**2 + mesh[0]**2 < r2 ,  r1 < mesh[1]**2 + mesh[0]**2) ] = 1;

	#getting bin image of pattern
	bpatBW = Im2bin(pat); #It seems DMD doesn't with this format only
	# bpat = sk.color.grey2rgb(rpat); #In case RGB format is need
	dummie = sk.img_as_bool(bpatBW); #to a boolean

	return dummie

diskpat = disk(mesh, origin, 10, diskp, profile, gap)

plt.figure(7)
plt.imshow(diskpat,cmap='rainbow')
plt.grid(b=None, which='both', axis='both')
saveIm(diskpat, 'Disk_Pattern.png')
dmd_prep(ref_im, 'Disk_Pattern.png', 'Disk_Pattern')


plt.show()         