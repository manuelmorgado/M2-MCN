
# coding: utf-8
#Calibrate Piezo Mirror


"""
Created: Feb. 18th, 2019
Last modification: Feb. 22th, 2019
MORGADO, Manuel (M2-MCN , QMAT fellow)
WINTERMANTEL, Tobias (PhD , QMAT fellow)
U. Strasbourg // ISIS // IPCMS
Supervisor. Prf. Dr. S. Whitlock

Calibration of the Piezo Mirror KIM101
Datasheet: https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=231
           https://www.thorlabs.com/drawings/ced5745d2177cb2-7230EF39-E8B8-0E79-27148B861B23040F/PIM05-Manual.pdf

Takes the data from Data Measuraments folder structure and fit the proper gaussians functions, compute the distance
between the minima of the gaussians and compute the difference of angle in the piezo mirror.           

"""

"""
TO DO List:

    - Fix setpath2dir()
    - Fix loadata()
    - Include dGauss() and pod_difM() which is necessary for double gaussians
    - Include minima, maxima, peaks and derivative analysis pks(), cvalues()
    - Include convolve method discv()
    - Include error bars
    - Fix plots for data and fit in the same figure
    - Fix details of figure plots
    - Fix order of lecture folders
    - Check x-axis missmatching with the .csv

"""
####| HERE IMPORTING LYBS |####

#importing lybraries
import numpy as np

import scipy as sp
from scipy import optimize
from scipy import signal
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox    # to add text to figure 
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True

import math as mt
import random

import glob
import time
import sys     #to add another active path to python to load the FreqConv class
import os
import datetime     #get time and date

####################################################################################################
####| HERE DOING DEFINITIONS |####

#noise function
def noise():
    noi = np.random.normal(0,0.03,400)
    return noi

#test functions
#line with random fluctuations (type y = m*x + b)
def rand_line(xval, m, b = 0):
    lineRand = [None]*len(xval);
    for i in range(len(xval)):
        lineRand[i] = np.random.randn(1)*(m*xval[i] + b);
    return lineRand

#normal gaussian
def Gauss(x, a, x0, sigma, c = 0):
    return a * np.exp(-((x - x0) / sigma)**2) + c


#gaussian with random fluctuations
def rand_gaus(xval, xval0, a, b, c ):
    gaussRand = [None]*len(xval);
    noi = np.asarray(noise());
    for i in range(len(xval)):
        gaussRand[i] = -(a * np.exp(-((xval[i]-xval0)/b)**2) + c) + noi[i]; #np.random.randn(1)* for noise
    return gaussRand

#doble gaussian with random fluctuations
def rand_dgaus(xval, xval01, a1, b1, c, xval02, a2, b2):
    gaussRand = [None]*len(xval);
    noi = np.asarray(noise());
    for i in range(len(xval)):
        gaussRand[i] = -(a1 * np.exp(-((xval[i]-xval01)/b1)**2) + a2 * np.exp(-((xval[i]-xval02)/b2)**2)  + c) + noi[i]; #np.random.randn(1)* for noise
    return gaussRand

#parabola with random fluctuations
def rand_parab(xval, a, b, c):
    paraRand = np.zeros(len(xval));
    for i in range(len(xval)):
        paraRand[i] =  (a*xval[i] **2 + b*xval[i] + c);
    return paraRand

#multiplot function for plot many lineshapes in a plot
def multiplot(list, title='Title'):
    plt.figure()
    for i in list:
        plt.plot(i)
        plt.title(title)
        plt.grid()
        
#function for discrete convolution of a list
def discv(datalst):
    cv = [None]*(len(datalst)-1);
    for i in range(len(datalst)-1):
        cv[i] = np.convolve(data_lst[i],data_lst[i+1]);
    return cv

#function for compute difference between functions in a list
def pos_dif(func, axis, flst):
    fg_lst = [];
    xvalues = axis;    

    #loop for the different curves
    for i in range(len(flst)):
        yvalues =  np.asarray(flst[i]);

        #gaussian fitting
        #weighted arithmetic mean
        mean = sum(xvalues * yvalues) / sum(yvalues);
        # mean = np.mean(yvalues);
        sigma = np.sqrt(sum(yvalues * (xvalues - mean)**2)/ sum(yvalues));   
        popt, pcov = sp.optimize.curve_fit(Gauss, xvalues, yvalues, p0=[(min(yvalues)-max(yvalues)), mean, sigma, min(yvalues)]);

        #minimum detecting
        fg = Gauss(xvalues,popt[0],popt[1],popt[2],popt[3]);
        y_min = np.asarray(np.where(fg == fg.min()));
        fg_lst.append(fg);
        
        #storage of list of mins
        min_lst.append(y_min);

    print('Plot... ')
    multiplot(flst)
    multiplot(fg_lst, 'Fitting curve')

    min_dif = [];
    for i, j in zip(min_lst[:-1], min_lst[1:]):
        dif = np.abs(j-i)
        min_dif.append(dif);

    #averaging the differences
    ave = np.mean(min_dif)

    return (min_dif, ave, fg_lst) #returns the list of minima and the average of the difference

#function for transformation from linear displacement to angular displacement
def dist2ang(distlst,fdist,yPMdist,pdist,N):
    """
    Transform of linear distance differences into angle differences
    
    distlst: distance to be convert (x->theta)
    fdist: focal distance
    yPMdist: y-axis direction to Piezo Mirror distance
    pdist: distance to plane that contain distlst
    N: number of functions
    """
    ang_lst = np.arctan(np.asarray(distlst)*fdist*yPMdist/(pdist-fdist))
    
    #plot of calibration of x- and theta-displacement
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure()
    plt.plot(np.reshape(distlst,N-1), marker='o')
    plt.title(r'Differences $\Delta X$')
    plt.xlabel(r'$\#$')
    plt.grid()
    plt.ylabel(r'$\Delta X$')

    plt.figure()
    plt.plot(np.reshape(ang_lst, N-1), marker='o')
    plt.title(r'$\theta(x)$')
    plt.xlabel('x')
    plt.ylabel(r'$\theta=tag^{-1}(x)$')
    plt.grid()
    plt.show()
    return ang_lst

#function for set path direction to files with data
def setpath2dir(): ##Need some fix
    plt.close('all')   #close all open figure windows

    #load the class which allows the conversion of voltages applied to the AOM into actual frequencies [MHz]
    last_AOMcal = '20180530'       #date of the AOMcalibration.pickle file to load

    #import needed analysis script from the python analysis scripts directory
    pyscript_test1 = r'C:\Users\Tobia\Dropbox\PhD\Python'   #path to the "parent" folder of the AOM calibration folder "DATE_AOMcalib_39K" 
    pyscript_test2 = r'X:\AnalysisScripts'         #path to the "parent" folder of the AOM calibration folder "DATE_AOMcalib_39K"
    pyscript_test3 = '/Users/Manuel/Desktop/'

    if os.path.isdir(pyscript_test1):
        pyscript_fold = pyscript_test1     #path on Tobias' Dell laptop
        print('Set folder parth to: ', pyscript_fold)
    elif os.path.isdir(pyscript_test2):       #path on experimental analysis computer in the lab
        pyscript_fold = pyscript_test2
        print('Set folder parth to: ', pyscript_fold)
    elif os.path.isdir(pyscript_test3):       #path on Manuel's Macbook laptop
        pyscript_fold = pyscript_test3
        print('Set folder parth to: ', pyscript_fold)
    else:
        print('Folder of Python analysis scipts was not found.')

    sys.path.insert(0, pyscript_fold)
    sys.path.insert(0, os.path.join(pyscript_fold, last_AOMcal + "_AOMcalib_39K"))      # dd this path to active python's paths 

    from pltFuns_20180530 import csv_import

    curr_dir = os.getcwd()
    # parent_dir = os.path.split(curr_dir)[0] #windows-like computers
    parent_dir = os.path.split(curr_dir)[0]+'/'+os.path.split(curr_dir)[1]+'/'
    return parent_dir

#function for load data from .csv files
def loadata(parent_dir): ##Need some fix
    
    """
    Possible variables to change

     To do:  
        adjust varX and varZ as variables which should be plotted. 
        Put the folders you want to plot into foldNames. 
        Optional legend names in legendtitles.
    """
    varX = 'yPiezoMirror'    #x variable to plot
    varZ = 'PeakOD'         #z variable to plot

    foldNames = [r'2019-02-15_12-58-22_xPiezoMirror', r'2019-02-15_13-08-21_yPiezoMirror',r'2019-02-15_13-22-15_xPiezoMirror']

    legendtitles = []   #strings for the legend, optional; if empty, otherwise folder names are taken

    if legendtitles == []:
        legendtitles=foldNames

    datX = csv_import(os.path.join(parent_dir,foldNames[0],'data1.csv'),varX)
    datZ = csv_import(os.path.join(parent_dir,foldNames[0],'data1.csv'),varZ)

    #initializing list for data calibration
    piezo_posY = [];
    peakOD_lst = [];

    for kk in foldNames[1:]:
        datX = csv_import(os.path.join(parent_dir,kk,'data1.csv'),varX)
        datZ = csv_import(os.path.join(parent_dir,kk,'data1.csv'),varZ)

        piezo_posY.append(datZ);
        peakOD_lst.append(datX);
    return (piezo_posY, peakOD_lst)

def csv_import(csvfile, variable):
    """Import data of a certain variable from a csv file from within the folder of this script:
       variabledata = csv_import(csvfile, variable)
       csvfile = string of csv file name to import, e.g. 'data1.csv'
       variable = string of the variable name within the csv to import, e.g. 'PeakOD'   
    """
    # typically peakOD = data1.csv and fluorescence = data234.csv
    try:                                                                    
        measdata = np.genfromtxt(csvfile, dtype=None, delimiter='\t,', names=True)
        try: 
            res = np.nan_to_num(measdata[variable])
        except:
            res = 0
            print('ERROR: variable %s name can not be found in the csv data.' % (variable))
    except:
        measdata = 0
        res = 'Error'
        print('ERROR: file %s could not be opened.' % (csvfile))
    return res

#example for setpath2dir() + loadata()
# parent_dir = setpath2dir()
# x_axis, data_lst = loadata(parent_dir)

####################################################################################################
####| HERE SET DIRECTORY |####


plt.close('all')   #close all open figure windows

#load the class which allows the conversion of voltages applied to the AOM into actual frequencies [MHz]
# last_AOMcal = '20180530'       #date of the AOMcalibration.pickle file to load

#import needed analysis script from the python analysis scripts directory
pyscript_test1 = r'C:\Users\Tobia\Dropbox\PhD\Python'   #path to the "parent" folder of the AOM calibration folder "DATE_AOMcalib_39K" 
pyscript_test2 = r'X:\AnalysisScripts'         #path to the "parent" folder of the AOM calibration folder "DATE_AOMcalib_39K"
pyscript_test3 = '/Users/Manuel/Desktop/' #path to the "parent" folder of the AOM calibration folder "DATE_AOMcalib_39K"

if os.path.isdir(pyscript_test1):
    pyscript_fold = pyscript_test1     #path on Tobias' Dell laptop
    print('Set folder parth to: ', pyscript_fold)
elif os.path.isdir(pyscript_test2):       #path on experimental analysis computer in the lab
    pyscript_fold = pyscript_test2
    print('Set folder parth to: ', pyscript_fold)
elif os.path.isdir(pyscript_test3):       #path on Manuel's Macbook laptop
    pyscript_fold = pyscript_test3
    print('Set folder parth to: ', pyscript_fold)
else:
    print('Folder of Python analysis scipts was not found.')

sys.path.insert(0, pyscript_fold)
# sys.path.insert(0, os.path.join(pyscript_fold, last_AOMcal + "_AOMcalib_39K"))      # dd this path to active python's paths 

curr_dir = os.getcwd()
# parent_dir = os.path.split(curr_dir)[0] #windows-like computers
parent_dir = os.path.split(curr_dir)[0]+'/'+os.path.split(curr_dir)[1]+'/'
print('es>:' , parent_dir)

####################################################################################################
####| HERE LOAD DATA |####

#defining x-axis for test functions

varX = 'yPiezoMirror'    #x variable to plot
varZ = 'PeakOD'         #z variable to plot

foldNames = [r'2019-02-14_18-08-52_yPiezoMirror',r'2019-02-14_17-06-51_yPiezoMirror',r'2019-02-14_16-52-39_yPiezoMirror',r'2019-02-14_16-39-33_yPiezoMirror',r'2019-02-14_16-29-23_yPiezoMirror',r'2019-02-14_16-19-20_yPiezoMirror']

legendtitles = []   #strings for the legend, optional; if empty, otherwise folder names are taken

if legendtitles == []:
    legendtitles=foldNames

datX = csv_import(os.path.join(parent_dir,foldNames[0],'data1.csv'),varX)
datZ = csv_import(os.path.join(parent_dir,foldNames[0],'data1.csv'),varZ)

#initializing list for data calibration
piezo_posY = [];
peakOD_lst = [];

#for loop for import and save data
for kk in foldNames:
    
    print('Loading... '+kk)

    datX = csv_import(os.path.join(parent_dir,kk,'data1.csv'),varX)
    datZ = csv_import(os.path.join(parent_dir,kk,'data1.csv'),varZ)

    piezo_posY.append(datZ);
    peakOD_lst.append(datX);

#creating list with data for calibration   
x_axis = np.asarray(peakOD_lst[0]);
data_lst = piezo_posY

multiplot(data_lst)
plt.show()
N = len(data_lst);

####################################################################################################
####| HERE DOING CALIBRATION |####

#computing difference between functions in data_lst
min_lst = []; # minima list
min_dif, average, fg_lst = pos_dif(Gauss, x_axis, data_lst); #difference

#computing transformation of difference to angular displacement difference
f_dist = 2.0; #mm
yPM_dist = 1.0; #mm
p_dist = 1.0; #mm
min_dif_ang = dist2ang(min_dif, f_dist, yPM_dist, p_dist, N);

plt.show()
