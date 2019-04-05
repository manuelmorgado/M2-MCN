#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Mar. 25th, 2019
Last modification: Mar. 13th, 2019
MORGADO, Manuel (M2-MCN , QMAT fellow)
U. Strasbourg // ISIS // IPCMS
Supervisor. Prf. Dr. S. Whitlock

FRESNEL DIFFRACTION IMAGE PATTERN

"""
####################################################################################################

"""
Script that creates a Fresnel ring pattern from LightPipes library for emulation of optics.

TO DO List:

    - 
"""

####################################################################################################
####| HERE IMPORTING LYBS |####
try:
	from LightPipes import *
	import matplotlib.pyplot as plt
except:
	print('Error importing packages.')

####################################################################################################
####| DEFINING SOME ASPECTS OF THE FIELD |####
Field = Begin(20*mm, 1*um, 256); #creates the field (laser)
Field = CircAperture(5*mm, 0, 0, Field); #creates the obstacle
Field = Forvard(1*m, Field); #fft
I = Intensity(1000,Field); #defines the intensity of the field (laser)

x = [];

####################################################################################################
####| CREATING THE DIFRACTION PATTERN |####
for i in range(256):
    x.append((-20*mm/2+i*20*mm/256)/mm);

####################################################################################################
####| PLOTS |####
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(I,cmap='rainbow'); ax1.axis('off')
ax2.plot(x,I[128]);ax2.set_xlabel('x [mm]');ax2.set_ylabel('Intensity [a.u.]')
ax2.grid('on')

plt.show()
