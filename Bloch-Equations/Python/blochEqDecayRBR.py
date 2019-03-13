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

#multiplot function for plot many lineshapes in a plot
def multiplot(axis,lst, title='Title', ax=0):
    if ax==0:
        plt.figure()
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
    # Omega = 1.46446*163*2*np.pi*1000; #[KHz]
    # g = 5.6*2*np.pi*1000; #[KHz]
    # noi_var = 0.0001; 
    # de = 914.0*2*np.pi; #[Hz]
    # Delta = -12.0*2*np.pi*1000000; #[MHz]
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
omega : Rabi's frequency or usually denoted as Omega_R (float)[UNITS]
g_decay : decay rate from r->x where x is some other stay of decay (float array) [UNITS] 
gamma : it is decay rate from r->g (float array) [UNITS]
n : the number of thermal photons (float) [UNITS]
delta : detuning (float array) [UNITS]
"""

omega = np.complex(input("Introduce a value for the Rabi frequency [MHz]: ")); #request a Rabi frequency's value
g_decay = [0.01*omega, 0.25*omega, 0.75*omega, 15*omega]; #set values of decay rate

# #to be use for an scan on omega->g_decay
# omega = np.linspace(0,5,10, dtype=np.complex_);
# g_decay = np.linspace(0,3,500);

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
rho0 = [0.0, 0.0, 1.0];
t = np.linspace(0,3,500);
rho11 = list(); rho12 = list(); rho22 = list(); 
rhoT = list();
t_minima = list();
i=0; #counter in the number of atoms

frequ = list();
# minima = list();

####################################################################################################
####| HERE DOING THE CALCULATIONS OF THE FREQUENCY AND SOLVING OBE WITH EXTRA DECAY AND N-ENSEMBLE|#

# #loop for the different values of gamma, delta, omega and gamma decay
# for k in range(4):

#     n_ensemble = 20;
#     om = np.sqrt(n_ensemble)*omega;
#     gd = g_decay[k];
#     # print(om, '|| ', type(om))
#     # print(gd, '||', type(gd))


#     g = gamma[3];
#     d = delta[3];

#     #solving the differential equation
#     rho, infodict = odeintz(bloch_eq, rho0, t, full_output=True);

#     #storing the k-th results for r12 and r22
#     r11 = rho[:,0];
#     r12 = rho[:,1];
#     r22 = rho[:,2];

#     #stacking in the data
#     rho11.append(r11);
#     rho12.append(r12);
#     rho22.append(r22);

#     #rho22 at certain time
#     # rhoT.append(r22[350]);

# #ploting
# plt.figure('rho22 (variable: = )')
# f1, = plt.plot(t, rho22[0], 'r-', label=r'\Gamma = 4\Omega_R; \delta =  0; \gamma = 0.01\Omega_R')
# f2, = plt.plot(t, rho22[1], 'b-', label=r'\Gamma = 4\Omega_R; \delta =  0; \gamma = 0.25\Omega_R')
# f3, = plt.plot(t, rho22[2], 'g-', label=r'\Gamma = 4\Omega_R; \delta =  0; \gamma = 0.75\Omega_R')
# f4, = plt.plot(t, rho22[3], '-', label=r'\Gamma = 4\Omega_R; \delta =  0; \gamma = 15\Omega_R')
# plt.legend(handler_map={f1: HandlerLine2D(numpoints=4)})
# plt.title(r'$\rho_{22}$ with decay term. $\Omega_R=1.5$')
# plt.ylabel(r'\rho_{22}(t)')
# plt.xlabel('t')
# plt.grid()


#loop for the different values of n_ensemble
n_ensemble = np.linspace(0,100,101);
for natoms in n_ensemble:

    om = np.sqrt(natoms)*omega;
    gd = g_decay[2];
    g = gamma[3];
    d = delta[3];

    #solving the differential equation
    rho, infodict = odeintz(bloch_eq, rho0, t, full_output=True);

    #storing the natoms-th results for r12 and r22
    r11 = rho[:,0];
    r12 = rho[:,1];
    r22 = rho[:,2];

    #fit for the Oscillatory decay functions
    """
    This doesn't work with a function like: a*exp(-b*t)*cos(w*t+c)+d ,
    even though Plot[(2 *x* exp(-x*3) * cos(10*x)/(x ))+3e^(-x)] for x in [0 to 4]
    seems to be very similar by using Worlfram.

    We then use the eq.(6) in : where is now going to be our function OscDec(). 

    OscDec(x, fr, g, noi_var, de, d, phi)
    """

    popt, pcov = sp.optimize.curve_fit(OscDec, t, np.real(r22), p0=[((np.real(om)**2)/(np.real(g)**2+4*np.real(d)**2)),  np.real(g)+np.real(gd),  0.0,  0.0,  np.real(d),  0.0]);
    f_om = np.sqrt(popt[0]*(np.real(g)**2 + 4*np.real(d)**2));
    frequ.append(f_om/omega);
    # plt.figure('Fits')
    # plt.plot(t,OscDec(t, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]))

    #finding the minima
    find_min = argrelextrema(r22, np.less);
    t_minima.append(t[find_min]);

    #stacking in the data
    rho11.append(r11);
    rho12.append(r12);
    rho22.append(r22);

    i+=1; #counting

plt.figure('Square Root behavior (By fitting)')
plt.plot(frequ)
plt.ylabel(r'$\frac{\Omega}{\Omega_R}$')
plt.xlabel('\# Atoms')

avedif = ave_dif(t_minima);
plt.figure('Square Root behavior (By average of Difference)')
plt.plot(n_ensemble, avedif)
# plt.title(r'$\rho_{22}$ with decay term. $\Omega_R=1.75$')
plt.ylabel(r'$\frac{\Omega}{\Omega_R}$')
plt.xlabel('\# Atoms')
plt.ylim([0.0, 20])
plt.xlim([0.0, 100])

#ploting
multiplot(t,rho22, r'$\rho_{22}$ with decay term. $\Omega_R=$ %.2f ' %(np.real(omega)))
plt.ylabel(r'$\frac{\rho_{22}(t)}{\rho_{22}(0)} $')
plt.xlabel('t [ms]')


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