"""
Solving the Bloch equations WITH DECAY
by using the atomic ensembles of N
Rydberg excitations.
"""


"""
Created: Feb. 28th, 2019
Last modification: Mar. 1st, 2019
MORGADO, Manuel (M2-MCN , QMAT fellow)
U. Strasbourg // ISIS // IPCMS
Supervisor. Prf. Dr. S. Whitlock

This solve the Bloch optical equation for a given value of the parameters:
gamma, n, rabi_freq and delta (decay rate, number of thermal photons, Rabi's
frequency) by using the scipy built-in command for solve differential equations
such as odeint from scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

Moreover, the plot for the different values of the decay and detuning haven being
perform.

(gamma = 0.0 ; delta = 0.0
gamma = 0.0 ; delta = 4*Omega_R
gamma = 0.2*Omega_R ; delta = 0.0
gamma = 2*Omega_R ; delta = 0.0)

Additionally, we introduce a term of decay for the particles in the states |rr> and |rg> ^ |gr>, 
meaning the loss of particles. In this scenario the constrain of the trace of rho doesn't hold.

In this case, we consider an ensemble of N-Rydberg atoms, the value of the coupling (previously the Rabi frequency) 
is now in function of the number of atoms in the ensemble, due to the coupling to the coherent superposition
of the Rydberg excitation (N)^{-1/2}*(\Sum{|ggr...g> + |ggg...r...g> + |ggg...g...rg> + ... + |ggg...g...gr>})
driving to a behavior of the frequency in terms of √N.

NOTE: notation between |1> and |2> it is exchanged by |g> and |r> during the script.
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
import scipy as sp

from scipy.integrate import odeint
from scipy import optimize
from scipy.signal import argrelextrema

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

#definition of solver of complex diferential equations system based in 'oint' (scipy) for 
# solve the real part of the system.
'''
Internal function for odeint in complex DE took from: 
https://stackoverflow.com/questions/19910189/scipy-odeint-with-complex-initial-values 
(slightly modified)
'''
def odeintz(func, z0, t, **kwargs):
    """
    An odeint-like function for complex valued differential equations.
    """

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args];
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1);

    def realfunc(x, t, *args):
        z = x.view(np.complex128);
        dzdt = func(z, t, *args);
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs);

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128);
        infodict = result[1];
        return z, infodict
    else:
        z = result.view(np.complex128);
        return z

#bloch optical equations
def bloch_eq(rho,t):
    r11 = rho[0];
    r12 = rho[1];
    r22 = rho[2];

    # dr11dt = ((-1j)*(np.conjugate(om))*(np.conjugate(r12))) + ((1j)*(om)*(r12)) + ((g)*(r22)) + ((n)*(g)*(r22)) - ((n)*(g)*(r11));
    dr11dt = (-1j)*(np.conjugate(om))*(np.conjugate(r12)) + (1j)*(om)*(r12) + (g)*(n+1)*(r22) - (n)*(g)*(r11);
    #Adding a decay term of the population for r22
    # dr12dt = ((-1)*(1j)*(d)*(r12)) - ((1j)*(np.conjugate(om))*(r22-r11)) - (((0.5)*(g)*((2*n)+1)*(r12))) - ((1j)*(g_decay)*(r12));
    dr12dt = (r12)*((-1j)*(d) - (0.5)*(g)*(2*n+1) - (1)*(gd)) - (1j)*(np.conjugate(om))*(r22-r11);
    #Now the r11+r22=1 doesnt hold anymore
    # dr22dt = ((1j)*(np.conjugate(om))*(np.conjugate(r12))) - ((1j)*(om)*(r12)) - ((n)*(g)*(r22)) - ((g)*(r22)) + ((n)*(g)*(r11)) - ((1j)*(g_decay)*(r22));
    dr22dt = (1j)*(np.conjugate(om))*(np.conjugate(r12)) - (1j)*(om)*(r12) - ( (n)*(g) + (g) + ((1)*(gd)) )*(r22) + (n)*(g)*(r11);
    return [dr11dt, dr12dt, dr22dt]

#bloch optical equations, interacting ensembles
def bloch_eq2(rho, t):
    r11A = rho[0];
    r12A = rho[1];
    r22A = rho[2];

    r11B = rho[3];
    r12B = rho[4];
    r22B = rho[5];

    dr11dtA = (-1j)*(np.conjugate(om))*(np.conjugate(r12A)) + (1j)*(om)*(r12A) + (g)*(n+1)*(r22A) - (n)*(g)*(r11A);
    dr12dtA = (r12A)*((-1j)*(d) - (0.5)*(g)*(2*n+1) - (1)*(gd)) - (1j)*(np.conjugate(om))*(r22A-r11A);
    dr22dtA = (1j)*(np.conjugate(om))*(np.conjugate(r12A)) - (1j)*(om)*(r12A) - ( (n)*(g) + (g) + ((1)*(gd)) )*(r22A) + (n)*(g)*(r11A) + Vab*r22A*r22B;
    
    dr11dtB = (-1j)*(np.conjugate(om))*(np.conjugate(r12B)) + (1j)*(om)*(r12B) + (g)*(n+1)*(r22B) - (n)*(g)*(r11B);
    dr12dtB = (r12B)*((-1j)*(d) - (0.5)*(g)*(2*n+1) - (1)*(gd)) - (1j)*(np.conjugate(om))*(r22B-r11B);
    dr22dtB = (1j)*(np.conjugate(om))*(np.conjugate(r12B)) - (1j)*(om)*(r12B) - ( (n)*(g) + (g) + ((1)*(gd)) )*(r22B) + (n)*(g)*(r11B) - Vab*r22B*r22A;

    return [dr11dtA, dr12dtA, dr22dtA, dr11dtB, dr12dtB, dr22dtB]

#multiplot function for plot many lineshapes in a plot
def multiplot(axis,lst, title='Title', ax=0):
    if ax==0:
        # plt.figure()
        i=0;
        for j in lst:
            plt.grid()
            f1 , = plt.plot(axis,j,label='\#Atoms = %.0f' %(float(n_ensemble[i])))
            # plt.legend(handler_map={f1: HandlerLine2D(numpoints=len(lst))})
            plt.title(title)
            i+=1;

    elif ax!=0:
        for i,j in zip(axis,lst):
            ax.plot(i,j)
            ax.set_title(title)
            ax.grid()

#function that compute the average of difference
def ave_dif(lst):
    cum = list(); avedif = list();
    for i in lst:
        sample = i;
        for j in range(len(sample)):
            try:
                dif = np.abs(sample[j+1]-sample[j]);
                cum.append(dif);
            except:
                pass;
        mean_cum = np.mean(cum);
        avedif.append(2*np.pi/(omega*mean_cum));
    return avedif

#function of oscillatory + exp decay [taken from Alda's paper]
def OscDec(x, fr, g, noi_var, de, dh, phi):
    """
    Equation equation (6) from PhysRevLett.122.053601 

    fr: relative population of the Rydberg state
    g: decay rate from |r> to |s> 
    noi_var: noise variance
    de:  Ramsey detuning * 2π (detuning of the rf field from the clock transition)
    d:  detuning of the Rydberg laser coupling 
    phi: phase shift ϕ caused by the detuning during the two π=2 pulses
    """
    #REFERENCE VALUES FROM ARIAS ET AL 
    # Omega = 1.46446*163*2*np.pi*1000; #[Hz]
    # g = 5.6*2*np.pi*1000; #[Hz]
    # noi_var = 0.0001; 
    # de = 914.0*2*np.pi; #[Hz]
    # Delta = -12.0*2*np.pi*1000000; #[Hz]
    # phi = np.degrees(0.55);
    # fr = Omega^2/(g^2 + 4*Delta^2);

    a = fr*g*10e12;
    b = fr*g*10e12 + noi_var;
    c = de*1000;
    d = fr*dh;
    e = phi;
    return ( 0.25*(1 + np.exp(-(a*x)) + 2*np.exp(-(0.5*(a + b)*x))*np.cos((c + d)*x + e)))

####################################################################################################
####| HERE DOING SOME GENERAL SETTINGS |####

#general parameters
"""
omega : Rabi's frequency or usually denoted as Omega_R (float)[KHz]
g_decay : decay rate from r->x where x is some other stay of decay (float array) [KHz] 
gamma : it is decay rate from r->g (float array) [KHz]
n : the number of thermal photons (float) [UNITS]
delta : detuning (float array) [Hz]
"""

omega = np.complex(input("Introduce a value for the Rabi frequency [KHz]: ")); #request a Rabi frequency's value [KHz]
g_decay = [0.01*omega, 0.25*omega, 0.75*omega, 15*omega]; #set values of decay rate [KHz]

gamma = [0.0, 0.0, 0.2*omega, 2*omega];
delta = [0.0, 4*omega, 0.0, 0.0];
n = 0.0;

#differential equation parameters
'''
t : linespace for the horizontal axis (time parameter)
rho : density matrix [(r11, r12);(r21,222)]

With initial conditions where the system initially is in the ground-state (|g>)
r11[0]=0; r12[0]=0; r21[0]=0; r22[0]=1; <-- Very important
'''
rho01 = [0.0, 0.0, 1.0];
rho02 = [0.0, 0.0, 1.0];
rho01.extend(rho02)
t = np.linspace(0,3,500);

rho11A = list(); rho12A = list(); rho22A = list(); 
rho11B = list(); rho12B = list(); rho22B = list(); 

rhoT = list();
t_minima = list();

frequ = list();

####################################################################################################
####| HERE DOING THE CALCULATIONS OF THE FREQUENCY AND SOLVING OBE WITH EXTRA DECAY AND N-ENSEMBLE|#

#loop for the different values of n_ensemble
n_ensemble = np.linspace(15,15,1);
for natoms in n_ensemble:

    om = np.sqrt(natoms)*omega;
    gd = g_decay[2];
    g = gamma[3];
    d = delta[3];
    Vab = 0.2;

    #solving the differential equation
    rho, infodict = odeintz(bloch_eq2, rho01, t, full_output=True);

    #storing the natoms-th results for r12 and r22
    r11A = rho[:,0];
    r12A = rho[:,1];
    r22A = rho[:,2];

    r11B = rho[:,3];
    r12B = rho[:,4];
    r22B = rho[:,5];

    #stacking in the data
    rho11A.append(r11A);
    rho12A.append(r12A);
    rho22A.append(r22A);
    rho11B.append(r11B);
    rho12B.append(r12B);
    rho22B.append(r22B);

#ploting
plt.figure()
multiplot(t,rho22A)
multiplot(t,rho22B)
# multiplot(t,rho22A)#, r'$\rho_{22}$ with decay term. $\Omega_R=$ %.2f ' %(np.real(omega)))
# plt.ylabel(r'$\frac{\rho_{22}(t)}{\rho_{22}(0)} $')
# plt.xlabel('t [ms]')


# fig = plt.figure('rho22 (omega, gdecay) @ t=1.75')
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(np.real(omega), np.real(g_decay), np.real(rhoT))

# plt.figure('rho12')
# g1, = plt.plot(t, rho12[0], 'r-', label=r'\Gamma = 0; \delta =  0')
# g2, = plt.plot(t, rho12[1], 'b-', label=r'\Gamma = 0; \delta =  4\Omega_R')
# g3, = plt.plot(t, rho12[2], 'g-', label=r'\Gamma = 0.2\Omega_R; \delta = 0')
# g4, = plt.plot(t, rho12[3], '-', label=r'\Gamma = 2\Omega_R; \delta =  0')
# plt.legend(handler_map={g1: HandlerLine2D(numpoints=4)})
# # plt.legend(loc='best')
# plt.ylabel(r'\rho_{12}(t)')
# plt.xlabel('t')
# plt.grid()


plt.show()