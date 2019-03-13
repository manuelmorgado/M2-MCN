#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Jan. 28th, 2019
Last modification: Mar. 12th, 2019
SHEVATE, Sayali (Ph.D , QMAT fellow)
MORGADO, Manuel (M2-MCN , QMAT fellow)
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
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import colors
import scipy.misc
import sys
from scipy import ndimage
import skimage as sk
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
"""
resX = 1024;
resY = 768;
P0 = 0.5; 
sigX = 146;
sigY = sigX; 

squarePat = np.zeros((resY, resX), dtype = int);
ringPat = np.zeros((resY, resX), dtype = int);
xPat = np.zeros((resY, resX), dtype = int);

x = np.arange(-resX/2,resX/2);   
y = np.arange(-resY/2,resY/2);

apRad = 30;
space = 30;
light_shift_coord_x = 4;
light_shift_coord_y = 4;

n = 8;
angle = 45;
mesh = np.meshgrid(x,y);

radius = 50.0;
origin = [int(0), int(0)];

#function for save image
def saveIm(imagen, namef='image.png' ):
	scipy.misc.toimage(imagen, cmin=0, cmax=1).save(namef);

#function that exports the image to binary
def Im2bin(pat, namef='image.png'):
	npat = pat.astype('bool');
	saveIm(npat, namef);
	return npat

#function of the site generator
def site(mesh, i, j, pat):
        pat[np.logical_and(np.abs(mesh[0]-j) < apRad/2, np.abs(mesh[1]-i) < apRad/2)] = 1;

#function of the circular site generator
def Osite(mesh, i, j, pat):
        pat[np.logical_and( np.abs( (mesh[0]-j)**2 ) < apRad/2 - np.abs((mesh[1]-i)**2) , np.abs( (mesh[1]-i)**2 ) < apRad/2 - np.abs( (mesh[0]-j)**2 ))] = 1;
        # NOTE: circles are no perfect when rotates because rotate command does not allows define exact coordinate

#function that defines the center and the positions of each site
def ring(center, radius, nr):

	xpos = [];
	ypos = [];

	#defininf center and angle between spots
	ang = 2*np.pi/nr;
	origin = center;

	#loop over the spot location 
	for i in range(nr):
		x, y = [int(radius*np.cos(ang*i)) , int(radius*np.sin(ang*i))];
		xpos.append(x) ; ypos.append(y);
	return xpos, ypos, origin

#function of the ring pattern generator
def RingPattern(mesh, xpos, ypos, pat):
	
	#generates a site in each spot on ring
	for i, j in zip(xpos, ypos):
		Osite(mesh, i, j, pat);

	#getting bin image of pattern
	# bpat = Im2bin(pat); #It seems DMD doesn't with this format only
	bpat = sk.color.grey2rgb(pat);
	return bpat

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
	# bpat = Im2bin(rpat); #It seems DMD doesn't with this format only
	bpat = sk.color.grey2rgb(rpat);

	return bpat

#function of the cross patter generator
def xPattern(center):
	xpos = [center[0], 0, center[0]-space, center[0]+space, 0];
	ypos = [center[1], center[1]+space, 0, 0, center[1]-space];

	for i in range(len(xpos)):
			Osite(mesh, xpos[i], ypos[i], xPat);

	#rotating patter
	rpat = PatternRotation(xPat, 0);
	#getting bin image of pattern
	# bpat = Im2bin(rpat); #It seems DMD doesn't with this format only
	bpat = sk.color.grey2rgb(rpat);

	return bpat	
#function for rotate pattern
def PatternRotation(pat, angle):
	data_rot = ndimage.rotate(pat, angle, mode='constant', cval=0);

	return data_rot

#multishow function for imshow many images in a subplot [NOT WORKING]
def multishow(patlst, title='Title'):
	
	fig, axs = plt.subplots(int(len(patlst)/2), int(len(patlst)/2), sharex=True, sharey=True)
    
    #setting counters for xy subplot positions
	k=0;
	l=0;
	n=0;
	for j in patlst:

		if k % 2 == 0:
		    axs[k, n].imshow(j);
		else:
		    axs[k+1, n].imshow(j);

		#when reach the end line of the subplot, change to next one
		if l == int(len(patlst)/2):
			n+=1;

		k+=1; l+=1;

####################################################################################################
######CREATING PATTERNS || NEW ||

pat_lst = [];

xpos, ypos, center = ring([0.0, 0.0], radius, n);
sqLattice = SquarePattern(squarePat, angle);
rgLattice = RingPattern(mesh, xpos, ypos, ringPat);
xLattice = xPattern(origin);

# pat_lst.append(SquarePattern(squarePat, angle));
# pat_lst.append(RingPattern(mesh, xpos, ypos, ringPat));
# pat_lst.append(xPattern(origin))

plt.figure(1)
plt.imshow(sqLattice)
plt.grid(b=None, which='both', axis='both')
saveIm(sqLattice, 'Square_Pattern.png')

plt.figure(2)
plt.imshow(rgLattice)
plt.grid(b=None, which='both', axis='both')
saveIm(rgLattice, 'Ring_Pattern.png')

plt.figure(3)
plt.imshow(xLattice)
plt.grid(b=None, which='both', axis='both')
saveIm(xLattice, 'Cross_Pattern.png')


# multishow(pat_lst)

plt.show()         