import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp
import json
import pandas as pd
from cpymad.madx import Madx
import NAFFlib
from math import modf
from matplotlib import pyplot as plt
from scipy.stats import linregress
import math
from scipy.stats import norm
from scipy.stats import kde
import time
from scipy import stats
import scipy as sp
import abel
from abel.direct import direct_transform
from abel.tools.analytical import GaussianAnalytical

def json_from_madx(file,outputdir):
    mad = Madx()
    mad.option(echo=False)
    mad.call(file)
    mad.use(sequence="lhcb1")
    line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'],
                                deferred_expressions=True
                                )
    with open(outputdir+'line.json', 'w') as fid:
        json.dump(line.to_dict(), fid, cls=xo.JEncoder)

def emittance(x,px,delta,dx,dpx):
    x_np = np.array(x)
    px_np= np.array(px)
    x_np = np.array(x - delta*dx)
    px_np = np.array(px - delta*dpx)

    x_mean_2 = np.dot(x_np,x_np)/len(x)
    px_mean_2 = np.dot(px_np,px_np)/len(x)
    x_px_mean = np.dot(x_np,px_np)/len(x)

    emitt = np.sqrt(x_mean_2*px_mean_2-(x_px_mean)**2)

    return emitt

def set_crabs_IP5(line,elem,kk,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps):
    line.element_dict[elem].ksl[0] = mu_ksl[kk] + a_noise[kk][ii]
    line.element_dict[elem].ps = mu_ps[kk] + ph_noise[kk][ii]
    #print(elem,line.element_dict[elem].ksl[0], line.element_dict[elem].ps)

def set_crabs_IP1(line,elem,kk,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps):
    line.element_dict[elem].knl[0] = mu_knl[kk] + a_noise[kk][ii]
    line.element_dict[elem].pn = mu_pn[kk] + ph_noise[kk][ii]
    #print(elem,line.element_dict[elem].knl[0], line.element_dict[elem].pn)

def set_INJ_crabs_IP5(line,elem,ksl):
    line.element_dict[elem].ksl[0] = ksl
    line.element_dict[elem].ps = 90
    line.element_dict[elem].knl[0] = 0
    line.element_dict[elem].pn = 0
    print(elem, line.element_dict[elem].knl[0], line.element_dict[elem].pn, line.element_dict[elem].ksl[0], line.element_dict[elem].ps)

def set_INJ_crabs_IP1(line,elem,knl):
    line.element_dict[elem].ksl[0] = 0
    line.element_dict[elem].ps = 0
    line.element_dict[elem].knl[0] = knl
    line.element_dict[elem].pn = 90
    print(elem, line.element_dict[elem].knl[0], line.element_dict[elem].pn, line.element_dict[elem].ksl[0], line.element_dict[elem].ps)


def _Cq(q):
    Gamma = math.gamma
    if q<0.99415629720:
        return (2.0*np.sqrt(np.pi))/((3.0-q)*np.sqrt(1-q))*(Gamma(1.0/(1.0-q)))/Gamma((3.0-q)/2.0/(1.0-q))
    elif q<1.005827 and q>0.99415629720:
        return np.sqrt(np.pi)
    elif (q>1.005827 and q<3.0):
        return (np.sqrt(np.pi)*Gamma((3.0-q)/2.0/(q-1.0)))/(np.sqrt(q-1.0)*Gamma(1.0/(q-1.0)))
    else:
        raise Exception('q<3')

def _eq(x,q):
    if (q!=1 and (1+(1-q)*x).all()>0):
        return (1+(1-q)*x)**(1/(1-q))
    elif q==1:
        return np.exp(x)
    else:
        return 0.0**(1.0/(1.0-q))

def qGauss(x, mu, q, b, A):
    return A*np.sqrt(b)/_Cq(q)*_eq(-b*(x-mu)**2,q)

def generate_phase_noise(ph_noise_rad = 0, ph_noise_mu = 0, scale_noise = 1, turns = 1):
    radtodeg = 180/np.pi
    ph_noise_sigma = ph_noise_rad*radtodeg*scale_noise#deg
    ph_mu, ph_sigma = ph_noise_mu, ph_noise_sigma #mean and standard deviation
    ph_noise = []
    for ii in range(16):
        ph_noise.append(np.random.normal(ph_mu, ph_sigma, turns)) #dipolar noise
    return ph_noise
def generate_amplitude_noise(a_noise_mu = 0, a_noise_sigma = 0, scale_noise = 1, turns = 1):
    a_noise_sigma = a_noise_sigma*scale_noise #ordine 10-8 dicevano alla presentazione
    a_mu, a_sigma = a_noise_mu, a_noise_sigma #mean and standard deviation
    a_noise = []
    for ii in range(16):
        a_noise.append(np.random.normal(0, a_sigma, turns)) #dipolar noise
    return a_noise

def produce_twiss_df(twiss):
    tw_df = pd.DataFrame({'name':twiss['name'],
                        's':twiss['s'],    
                        'betx':twiss['betx'],
                        'alfx':twiss['alfx'],
                        'gamx':twiss['gamx'],
                        'bety':twiss['bety'],
                        'alfy':twiss['alfy'],
                        'gamy':twiss['gamy'],
                        'dx':twiss['dx'],
                        'dpx':twiss['dpx'],
                        'dy':twiss['dy'],
                        'dpy':twiss['dpy']          
                        })
    return tw_df
def generate_q_gauss_bunch(q0 = 1, b0 = 0.5, A0 = 1, N_particles = 1):
    npoints = 20000
    xs = np.linspace(-10,10,npoints)
    #Number of particles to be tracked
    #Our q-Gauss distribution
    qgauss = qGauss(xs ,mu=0,q=q0,A=A0,b=b0)
    qgauss = np.nan_to_num(qgauss, 0)
    #We get to the r variable by using the inverse abel transform
    r = np.linspace(0.00001,10.00001,10000)
    #We transform the part of distribution at x>0
    abel_inverse_qgauss = direct_transform(qgauss[xs>0], dr=np.diff(r)[0], direction="inverse", correction=True)
    #We want the PDF in r, so we multiply by 2*pi*r
    abel_inverse_qgauss_r = 2*np.pi*r*abel_inverse_qgauss
    #Now we can calculate the probabilities
    r_probabilities= abel_inverse_qgauss_r/np.sum(abel_inverse_qgauss_r)
    r_probabilities = np.nan_to_num(r_probabilities)
    r_generated = np.random.choice(a=r,p=r_probabilities,size=N_particles)
    theta_generated  = np.random.uniform(0,2*np.pi,size=N_particles)
    x_coordinate = r_generated*np.cos(theta_generated)
    px_coordinate = r_generated*np.sin(theta_generated)
    
    return x_coordinate,px_coordinate

def get_normalized_phase_space(x, px, beta, alpha, scale):
    P = np.array([[np.sqrt(beta), 0],[-alpha/np.sqrt(beta), 1/np.sqrt(beta)]])
    X =np.array([x,px])
    return np.linalg.inv(P)@X*scale

def log_q(q,x):
    ''' 
    q = 1 -> log(x)
    x < 0 -> undefined
    q not 1, x > 0 -> x**(1-q) -1)/(1-q)
    ''' 
    if min(x) < 0:
        return 'error: x is smaller than 0'
    if q==1:
        return np.log(x)
    else:
        value = (x**(1-q) -1)/(1-q)
        return value
    
    
def box_muller_q_gauss(q, N):
    
    '''
    Generates 2 q-Gaussian distributions, but NOT independent
    '''
    q_prime = (1+q)/(3-q)
    U1 = np.random.uniform(size = N)
    U2 = np.random.uniform(size = N)
    
    R = np.sqrt(-2 * log_q(q_prime, U1))
    Theta = 2 * np.pi * U2
    X = R * np.cos(Theta)
    PX = R * np.sin(Theta)

    beta = 1/(3-q)
    
    return X, PX, beta
