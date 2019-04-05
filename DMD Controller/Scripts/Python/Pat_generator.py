#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Jan. 28th, 2019
Last modification: Mar. 13th, 2019
MORGADO, Manuel (M2-MCN , QMAT fellow)
SHEVATE, Sayali (Ph.D , QMAT fellow)
U. Strasbourg // ISIS // IPCMS
Supervisor. Prf. Dr. S. Whitlock

DMD Controller - DMD Patterns generator. 


"""
####################################################################################################

"""
TO DO List:

    - Fix the regularity of the shape of the spots when rotate
      (also optimizable with the experiments i.e exp. res.)
	- Fix multishow function
"""

####################################################################################################
####| HERE IMPORTING LYBS |####
try:
	import numpy as np 
	from matplotlib import pyplot as plt

	import scipy.misc
	from scipy import ndimage

	import skimage as sk

	from PIL import Image 
	from PIL.ExifTags import TAGS, GPSTAGS

	from ALP4 import *
	import time

	print('Succesfully import of packages.')
except:
	print('Error importing packages.')

####################################################################################################
####| HERE DOING DEFINITIONS |####
"""
resX : # of columns of DMD
resY : # of rows on DMD
P0 : gaussian beam power [W]
sigX : beam waist in x [px]
sigY : beam waist in y [px]

x : x-scale
y : y-scale
n : spots number in the lattice (nxn) 
	(NOTE: more interesting when it is grater than 5)

apRad : diameter of circular/square spot 
		(WARNING: for even values the spots are amorphous USE ODDS WHEN ROTATE)
space : centre to centre distance i.e. period (WARNING: scale is shfted by 0.5)
light_shift_coord_x : related to have anisotropic lattices in x-axis
light_shift_coord_y : related to have anisotropic lattices in y-axis

angle : rotation angle of the square lattice [degree]
mesh : define the mesh array
radius : radius of the ring pattern
origin : origin of the image (usually the center of the pattern)

ref_im : reference image's file name (DMD default file)

squarePat : pattern array for square patter made of many Osites
ringPat : pattern array for ring pattern made of many Osites
xPat : pattern array for cros pattern many Osites (e.g surface code -like)
varSite : pattern array for contracting circular Osite
varSite2 : pattern array for expanding circular Osite
sepsit : pattern array for two Osites separating each other
diskp : pattern array for the Fresnel disks difraction pattern
RoDp : pattern array for a Ring with Fresnel disks in each spot

profile : the sizes of the radius of the rings in Fresnel pattern
gap : between rings in Fresnel rings diffraction

"""
resX = 1024;
resY = 768;
P0 = 0.5; 
sigX = 146;
sigY = sigX; 

x = np.arange(-resX/2,resX/2);   
y = np.arange(-resY/2,resY/2);
n = 8;

apRad = 40;
space = 15;
light_shift_coord_x = 4;
light_shift_coord_y = 4;

angle = 45;
mesh = np.meshgrid(x,y);

radius = 40.0;
origin = [int(100), int(100)];

ref_im = 'refim.png';

squarePat = np.zeros((resY, resX), dtype = int);
ringPat = np.zeros((resY, resX), dtype = int);
xPat = np.zeros((resY, resX), dtype = int);
varSite = np.zeros((resY, resX), dtype = int);
varSite2 = np.zeros((resY, resX), dtype = int);
sepsit = np.zeros((resY, resX), dtype = int);
RoDp = np.zeros((resY, resX), dtype = int);

diskp = np.zeros((resY, resX), dtype = int);
profile = [ 12, 20, 35, 60, 90, 120];#, 160, 200, 250, 300,400, 600, 800, 1000, 1200]; #try to follow profile from .py
gap = 2;

#function for save image
def saveIm(imagen, namef='image.png' ):
	scipy.misc.toimage(imagen, cmin=0, cmax=1).save(namef);

#function that exports the image to binary
def Im2bin(pat, namef='image.png'):
	npat = pat.astype('bool');
	return npat

#function of the site generator
def site(mesh, i, j, pat):
        pat[np.logical_and(np.abs(mesh[0]-j) < apRad/2, np.abs(mesh[1]-i) < apRad/2)] = 1;

#function of the circular site generator
def Osite(mesh, i, j, pat):
        pat[np.logical_and( np.abs( (mesh[0]-j)**2 ) < apRad/2 - np.abs((mesh[1]-i)**2) , np.abs( (mesh[1]-i)**2 ) < apRad/2 - np.abs( (mesh[0]-j)**2 ))] = 1;
        # NOTE: circles are no perfect when rotates because rotate command does not allows define exact coordinate

#function of the circular site generator for a different values nad radius
def OsiteDif(mesh, i, j, pat, val,apRad):
        pat[np.logical_and( np.abs( (mesh[0]-j)**2 ) < apRad/2 - np.abs((mesh[1]-i)**2) , np.abs( (mesh[1]-i)**2 ) < apRad/2 - np.abs( (mesh[0]-j)**2 ))] = val;
        # NOTE: circles are no perfect when rotates because rotate command does not allows define exact coordinate

#function that defines the center and the positions of each site
def ring(center, radius, nr):

	xpos = [];
	ypos = [];

	#defining center and angle between spots
	ang = 2*np.pi/nr;
	origin = center;

	#loop over the spot location 
	for i in range(nr):
		x, y = [int(radius*np.cos(ang*i)) + origin[0], int(radius*np.sin(ang*i))+origin[1]];
		xpos.append(x) ; ypos.append(y);
	return xpos, ypos, origin

#function of the ring pattern generator
def RingPattern(mesh, xpos, ypos, pat):
	
	#generates a site in each spot on ring
	for i, j in zip(xpos, ypos):
		Osite(mesh, i, j, pat);

	#getting bin image of pattern
	bpatBW = Im2bin(pat); #It seems DMD doesn't with this format only
	# bpat = sk.color.grey2rgb(pat); #In case RGB format is need
	dummie = sk.img_as_bool(bpatBW); #to a boolean

	return dummie

#function of the square pattern generator
def SquarePattern(squarePat, angle):
	# variables: space, apRad, n, light_shift_coord_x, light_shift_coord_y, mesh, 

	if (space < apRad):
		sys.exit("Error: Invalid period_spacing. Please check.")
	else: 
		for i in range(-int((n-1)/2*space+apRad/2), int((n-1)/2*space+apRad/2), space):
			for j in range( -int( (n-1)/2*space + apRad/2 ), int((n-1)/2*space+apRad/2), space):
				if (abs((i+((n-1)/2*space+apRad/2))/space) == light_shift_coord_x and abs((j+((n-1)/2*space+apRad/2))/space) == light_shift_coord_y):
					Osite(mesh, i, j, squarePat);
				else:
					Osite(mesh, i, j, squarePat);

	#rotating patter
	rpat = PatternRotation(squarePat, angle);
	#getting bin image of pattern
	bpatBW = Im2bin(rpat); #It seems DMD doesn't with this format only
	# bpat = sk.color.grey2rgb(rpat); #In case RGB format is need
	dummie = sk.img_as_bool(bpatBW); #to a boolean

	return dummie

#function of the cross patter generator
def xPattern(mesh, center):
	xpos = [center[0], 0, center[0]-space, center[0]+space, 0];
	ypos = [center[1], center[1]+space, 0, 0, center[1]-space];

	for i in range(len(xpos)):
			Osite(mesh, xpos[i], ypos[i], xPat);

	#rotating patter
	rpat = PatternRotation(xPat, 0);
	
	#getting bin image of pattern
	bpatBW = Im2bin(rpat); 			  #It seems DMD doesn't with this format only
	# bpat = sk.color.grey2rgb(rpat); #In case RGB format is need
	dummie = sk.img_as_bool(bpatBW);  #to a boolean

	return dummie	

#function for rotate pattern
def PatternRotation(pat, angle):
	data_rot = ndimage.rotate(pat, angle, mode='constant', cval=0);

	return data_rot

#multishow function for imshow many images in a subplot [NOT WORKING]
def multishow(patlst, title='Title'):
	
	fig, axs = plt.subplots(int(len(patlst)/2), int(len(patlst)/2), sharex=True, sharey=True)
    
    #setting counters for xy subplot positions
	k = 0;
	l = 0;
	n = 0;
	for j in patlst:

		if k % 2 == 0:
		    axs[k, n].imshow(j);
		else:
		    axs[k+1, n].imshow(j);

		#when reach the end line of the subplot, change to next one
		if l == int(len(patlst)/2):
			n+=1;

		k+=1; l+=1;

#function to crop pics that are out of shape for the DMD
def imcrop(imagen, fshape):
	x,y = np.shape(imagen);
	difX = int(abs(fshape[0] - x)/2);
	difY = int(abs(fshape[1] - y)/2);
	box = (difX , difY, difX+fshape[0], difY+fshape[1]);

	crop = imagen.crop(box);
	imagen = crop;

	return imagen

#function that prepares the image obtained to a format readable by the DMD
def dmd_prep(refim, finalim, fname, angval=0, mode=False):

	pic1 = refim;	#reference image for extract the palette
	pic2 = finalim;	#image to be save

	image1 = Image.open(pic1);
	image2 = Image.open(pic2);

	pal1=image1.getpalette() #getting palette
	image2.putpalette(pal1)	 #assigning the reference palette in the final image's palette

	#check if there is any rotation in order to achieve a crop
	if angval != 0:
		image2 = imcrop(image2, (resX,resY));

	image2.save(fname + ".PNG", dpi=(72,72), bits=1);

#function that creates pattern that reduce size of a single Osite (sequence of images)
def expSite(mesh, Maxdia, pat):
	
	#list of frames
	frames = [];

	for dia in range(Maxdia):
		pat[np.logical_and( np.abs( (mesh[0]-0)**2 ) < dia/2 - np.abs((mesh[1]-0)**2) , np.abs( (mesh[1]-0)**2 ) < dia/2 - np.abs( (mesh[0]-0)**2 ))] = 1;
		saveIm(pat, 'Expanding_Site.png');
		dmd_prep(ref_im, 'Expanding_Site.png', 'Expanding_Site');

		#folding the frames
		frame = Image.open('Expanding_Site.png'); #might be necessary to increase the number of open files in the OS e.g MACOS : [CHECK: ulimit -a // CHANGE: ulimit -Sn 10000]
		name_frame = 'seq1_expan' + str(dia) + '.png'
		saveIm(frame, name_frame)
		frames.append(frame);

	#creating the animation
	"""
	Takes the list of frames and then save them in a .gif format, the duration is in terms of 10e-1 secs
	"""
	frames[0].save('animated_expand.gif', save_all = True, append_images = frames[1:], duration=10, oop=0);
	
	return frames

#function that creates pattern that expand size of a single Osite
def reSite(mesh, Maxdia, pat):

	#list of frames
	frames = [];

	#copy of the original pattern to be modify
	npat = np.copy(pat);

	for rad in range(Maxdia):

		#copy of the original pattern to be modify
		npat = np.copy(pat);

		#regresion way
		dia = Maxdia - rad;
		npat[np.logical_and( np.abs( (mesh[0]-0)**2 ) < dia/2 - np.abs((mesh[1]-0)**2) , np.abs( (mesh[1]-0)**2 ) < dia/2 - np.abs( (mesh[0]-0)**2 ))] = 1;
		
		#storing the current frame
		saveIm(npat, 'Reducing_Site.png');
		dmd_prep(ref_im, 'Reducing_Site.png', 'Reducing_Site');

		#folding the frames
		frame = Image.open('Reducing_Site.png'); #might be necessary to increase the number of open files in the OS e.g MACOS : [CHECK: ulimit -a // CHANGE: ulimit -Sn 10000]
		frames.append(frame);

	#creating the animation
	"""
	Takes the list of frames and then save them in a .gif format, the duration is in terms of 10e-1 secs
	"""
	frames[0].save('animated_reduce.gif', save_all = True, append_images = frames[1:], duration=10, oop=0);
	
	return frames

#function that creates pattern that separates 2 Osites
def sepOsites(mesh, center, Maxdist, pat):

	#list of frames
	frames = [];

	#copy of the original pattern to be modify
	npat = np.copy(pat);

	for dist in range(Maxdist):

		#regresion way of distance between sites
		dist = Maxdist - dist;

		xpos = [center[0], center[0]+dist];
		ypos = [center[1], center[1]];

		#copy of the original pattern to be modify
		npat = np.copy(pat);

		#creating sites
		for i in range(len(xpos)):
				# npat = np.copy(pat); #show this and hide the above one for driving
				Osite(mesh, xpos[i], ypos[i], npat);

		#rotating patter
		rpat = PatternRotation(npat, 0);
		
		#getting bin image of pattern
		bpatBW = Im2bin(rpat); 			  #It seems DMD doesn't with this format only
		# bpat = sk.color.grey2rgb(rpat); #In case RGB format is need
		dummie = sk.img_as_bool(bpatBW);  #to a boolean
		
		#storing the current frame
		saveIm(dummie, '2site_separation.png');
		dmd_prep(ref_im, '2site_separation.png', '2site_separation');

		#folding the frames
		frame = Image.open('2site_separation.png'); #might be necessary to increase the number of open files in the OS e.g MACOS : [CHECK: ulimit -a // CHANGE: ulimit -Sn 10000]
		frames.append(frame);

	#creating the animation
	"""
	Takes the list of frames and then save them in a .gif format, the duration is in terms of 10e-1 secs
	"""
	frames[0].save('animated_2site_separation.gif', save_all = True, append_images = frames[1:], duration=10, oop=0);

	return dummie

#function that creates disk pattern (fresnel pattern)
def disk2(mesh, center, pat, profile, gap):
    #copy of the original pattern to be modify
    npat = np.copy(pat);

    x0 = [center[0]];
    y0 = [center[1]];

    #for loop for the creation of the different rings
    for j in range(len(profile)):		

        r2j = float(x0[int(j)]) + float(profile[int(j)]);
        y0.append(r2j);

        r1jj = float(y0[int(j)]) + float(gap)*0.3;
        x0.append(r1jj);

        value = 0;

        for i in x0[::-1]:
            value = (value+1)%2;
            Rad = i*2;
            circ = OsiteDif(mesh, center[0], center[1], pat, value, Rad);

    #getting bin image of pattern
    bpatBW = Im2bin(pat); #It seems DMD doesn't with this format only
    # bpat = sk.color.grey2rgb(rpat); #In case RGB format is need
    dummie = sk.img_as_bool(bpatBW);

def ringOfDisks(mesh, xpos, ypos, RoDpat, profile, gap):
# def ringOfDisks(mesh, center, RoDpat, profile, gap):
	
	#generates a site in each spot on ring
	for i, j in zip(xpos, ypos):
		disk2(mesh, [i,j], RoDpat, profile, gap);

	#getting bin image of pattern
	bpatBW = Im2bin(RoDpat); #It seems DMD doesn't with this format only
	# bpat = sk.color.grey2rgb(pat); #In case RGB format is need
	dummie = sk.img_as_bool(bpatBW); #to a boolean

	return dummie

	# #creating the positions of the spots
	# xpos, ypos, origin = ring(center, radius, n);

	# #generates a site in each spot on ring
	# for i, j in zip(xpos, ypos):
	# 	spot = [i,j];
	# 	disk(mesh, spot, RoDpat, profile, gap);

	# #getting bin image of pattern
	# bpatBW = Im2bin(pat); #It seems DMD doesn't with this format only
	# # bpat = sk.color.grey2rgb(pat); #In case RGB format is need
	# dummie = sk.img_as_bool(bpatBW); #to a boolean

	# return dummie
####################################################################################################
######CREATING PATTERNS || NEW ||

# pat_lst = []; #list of pattern for use with multiplot #######??
try:
	print('Creating Ring Pattern: OK')
	xpos, ypos, center = ring([60.0, -60.0], radius, n);
	rgLattice = RingPattern(mesh, xpos, ypos, ringPat);
except:
		print('Creating Ring Pattern: ERROR')

try:
	print('Creating Square Pattern: OK')
	sqLattice = SquarePattern(squarePat, angle);
except:
	print('Creating Square Pattern: ERROR')

try:
	print('Creating Cross Pattern: OK')
	xLattice = xPattern(mesh, origin);
except:
	print('Creating Cross Pattern: ERROR')

try:
	print('Creating Reducing Pattern: OK')
	reSite = reSite(mesh, 70, varSite);
except:
	print('Creating Reducing Pattern: ERROR')

try:
	print('Creating Expanding Pattern: OK')
	expSite = expSite(mesh, 70, varSite2);
except:
	print('Creating Expanding Pattern: ERROR')

try:
	print('Creating 2 sites separation Pattern: OK')
	sepSite = sepOsites(mesh, origin, 70, sepsit);
except:
	print('Creating S2 sites separation Pattern: ERROR')

try:
	print('Creating Disk Pattern: OK')
	diskpat = disk(mesh, origin, diskp, profile, gap);
except:
	print('Creating Disk Pattern: ERROR')

# try:
print('Creating Ring of Disk Pattern: OK')
xpos, ypos, center = ring([0.0, 0.0], radius, n);
RoD = ringOfDisks(mesh, xpos, ypos, RoDp, profile, gap);
	# RoD = ringOfDisks(mesh, xpos, ypos, RoDp, profile, gap);
# except:
print('Creating Ring of Diks Pattern: ERROR')

# pat_lst.append(rgLattice);
# pat_lst.append(sqLattice);
# pat_lst.append(xLattice)
# pat_lst.append(reSite);
# pat_lst.append(expSite)
# pat_lst.append(sepSite)
# pat_lst.append(diskpat)

####################################################################################################
######DOING THE PLOTS AND SAVING FINAL IMAGES

try:
	saveIm(sqLattice, 'Square_Pattern.png')
	dmd_prep(ref_im, 'Square_Pattern.png', 'Square_Pattern', angle)
	print('Square Pattern saved: OK')
	plt.figure(1)
	plt.imshow(sqLattice)
	plt.grid(b=None, which='both', axis='both')
except:
	print('Square Pattern saved: ERROR')

try:
	saveIm(rgLattice, 'Ring_Pattern.png')
	dmd_prep(ref_im, 'Ring_Pattern.png', 'Ring_Pattern')
	print('Ring Pattern saved: OK')
	plt.figure(2)
	plt.imshow(rgLattice)
	plt.grid(b=None, which='both', axis='both')
except:
	print('Ring Pattern saved: ERROR')

try:
	saveIm(xLattice, 'Cross_Pattern.png')
	dmd_prep(ref_im, 'Cross_Pattern.png', 'Cross_Pattern')
	print('Cross Pattern saved: OK')
	plt.figure(3)
	plt.imshow(xLattice)
	plt.grid(b=None, which='both', axis='both')
except:
	print('Cross Pattern saved: ERROR')

try:
	saveIm(reSite, 'Reducing_Site.png')
	dmd_prep(ref_im, 'Reducing_Site.png', 'Reducing_Site')
	print('Reducing Site saved: OK')
	plt.figure(4)
	plt.imshow(xLattice)
	plt.grid(b=None, which='both', axis='both')
except:
	print('Reducing Site saved: ERROR')

try:
	saveIm(expSite, 'Expanding_Site.png')
	dmd_prep(ref_im, 'Expanding_Site.png', 'Expanding_Site')
	print('Expanding Site saved: OK')
	# plt.figure(5)
	# plt.imshow(expSite)
	# plt.grid(b=None, which='both', axis='both')
except:
	print('Expanding Site saved: ERROR')

try:
	saveIm(sepSite, '2site_separation.png')
	dmd_prep(ref_im, '2site_separation.png', '2site_separation')
	print('2 sites separation saved: OK')
	# plt.figure(6)
	# plt.imshow(sepSite)
	# plt.grid(b=None, which='both', axis='both')
except:
	print('2 sites separation saved: ERROR')

try:
	saveIm(diskpat, 'Disk_Pattern.png')
	dmd_prep(ref_im, 'Disk_Pattern.png', 'Disk_Pattern')
	print('Disk Pattern saved: OK')
	plt.figure(7)
	plt.imshow(diskpat)
	plt.grid(b=None, which='both', axis='both')
except:
	print('Disk Pattern saved: ERROR')

try:
	saveIm(RoD, 'Ring_of_Diks_Pattern.png')
	dmd_prep(ref_im, 'Ring_of_Diks_Pattern.png', 'Ring_of_Diks_Pattern')
	print('Ring of Diks Pattern saved: OK')
	plt.figure(8)
	plt.imshow(RoD)
	plt.grid(b=None, which='both', axis='both')
except:
	print('Disk Pattern saved: ERROR')

# multishow(pat_lst)

# plt.show()         

####################################################################################################
###### || DMD LOADING ||

# Load the Vialux .dll
# DMD = ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
# # Initialize the device
# DMD.Initialize()

# # Binary amplitude image (0 or 1)
# bitDepth = 1    
# # imgBlack = np.zeros([DMD.nSizeY,DMD.nSizeX])
# # imgWhite = np.ones([DMD.nSizeY,DMD.nSizeX])*(2**8-1)
# # imgSeq  = np.concatenate([imgBlack.ravel(),imgWhite.ravel()])

# imgSeq  = rgLattice;

# # Allocate the onboard memory for the image sequence
# DMD.SeqAlloc(nbImg = 2, bitDepth = bitDepth)
# # Send the image sequence as a 1D list/array/numpy array
# DMD.SeqPut(imgData = imgSeq)
# # Set image rate to 50 Hz
# DMD.SetTiming(illuminationTime = 20000)

# # Run the sequence in an infinite loop
# DMD.Run()

# time.sleep(10)

# # Stop the sequence display
# DMD.Halt()
# # Free the sequence from the onboard memory
# DMD.FreeSeq()
# # De-allocate the device
# DMD.Free()



