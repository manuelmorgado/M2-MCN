"""
Obtaining the energy of 1/c^6 (van der Waals energy) between the elements of two differents ensembles.
"""


"""
Created: Mar. 7th, 2019
Last modification: Mar. 8st, 2019
MORGADO, Manuel (M2-MCN , QMAT fellow)
S. Shevate (Ph.D , QMAT fellow)
U. Strasbourg // ISIS // IPCMS
Supervisor. Prf. Dr. S. Whitlock

This script set the positions of two different ensembles in the space (clouds) from a random normal 
generator and define the distance difference between elements of the atoms of both ensembles. Later, 
it show the distribution in histogram shape of the differences.
"""

####################################################################################################

"""
TO DO List:

    - 

"""

####################################################################################################
####| HERE IMPORTING LYBS |####

#importing lybraries 
import numpy as np
# import scipy as sp

# from scipy.integrate import odeint
# from scipy import optimize
# from scipy.signal import argrelextrema

import pandas as pd
import plotly.plotly as py
import cufflinks as cf

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.mplot3d import Axes3D

import os
os.system('cls' if os.name == 'nt' else 'clear') #clean terminal window

####################################################################################################
####| HERE DOING DEFINITIONS |####

#function that creates the 'random' normal positions of the ATOMS in the space (clouds of atoms)
def CloudAtoms(ax, centerXYZ=[0.0, 0.0, 0.0],  sdev=[0.0, 0.0, 0.0], Natoms=1):

	randPos = np.random.normal(centerXYZ, sdev, (Natoms,3));
	xPos, yPos, zPos = randPos[:,0], randPos[:,1], randPos[:,2];

	ax.scatter3D(xPos, yPos, zPos, c=zPos, cmap='viridis');

	return xPos, yPos, zPos

#function that creates arrays of clouds of atoms
def CloudsArray(Nclouds, centers, sdevs, Natoms):
	
	Positions = [];

	ax = plt.axes(projection='3d')

	for cloud in range(Nclouds):
		AtomsCloudPos = CloudAtoms(ax, centers[cloud], sdevs[cloud], Natoms[cloud]);
		Positions.append(AtomsCloudPos);
	
	# print(type(Positions), np.shape(Positions))

	return Positions 

def EuclideanDist(Positions, Nclouds, Natoms):
	
	Distances = [];
	
	Positions = np.asarray(Positions);
	shape = (3, Nclouds, Natoms);
	Positions.reshape(shape);

	cX1 , cY1, cZ1 = Positions[:,0,:], Positions[:,1,:], Positions[:,2,:] ;
	cX2 , cY2, cZ2 = Positions[:,0,:], Positions[:,1,:], Positions[:,2,:] ;

	print(cX1[8,9])

	for cloud1 in range(len(cX1[:,0])):
		for atom1 in range(len(cX1[0,:])):
			for cloud2 in range(len(cX1[:,0])):
				for atom2 in range(len(cX1[0,:])):

					xdifSqr = (cX1[cloud1][atom1]-cX2[cloud2][atom2])**2;
					ydifSqr = (cY1[cloud1][atom1]-cY2[cloud2][atom2])**2;
					zdifSqr = (cZ1[cloud1][atom1]-cZ2[cloud2][atom2])**2;
					distA1A2 = np.sqrt(xdifSqr + ydifSqr + zdifSqr);

					print('Cloud 1: ',cloud1, 'Atom 1: ',atom1 ,'Cloud 2: ',cloud2 ,'Atom 2: ',atom2)
					Distances.append(distA1A2);

	return Distances

def histog(data):
	
	cf.set_config_file(offline=False, world_readable=True, theme='pearl')

	df = pd.DataFrame({'Distances': data})

	df.head(2)

	df.iplot(kind='histogram', subplots=True, shape=(3, 1), filename='cufflinks/histogram-subplots')


Nclouds = 9;
Natoms = 10;

positions = CloudsArray(Nclouds, [[0,0,0], [1,1,1], [-1,1,1], [-1,-1,1], [1,-1,1], [1,1,-1], [-1,1,-1], [-1,-1,-1], [1,-1,-1]], [0.1, 0.2, 0.4, 0.1, 0.1, 0.1, 0.2, 0.24, 0.15], [Natoms, Natoms, Natoms, Natoms, Natoms, Natoms, Natoms, Natoms, Natoms]);
distances = EuclideanDist(positions, Nclouds, Natoms);

plt.figure('Histogram')
plt.hist(distances)

plt.show()



