#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
from statistics import NormalDist
import scipy
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import os
import scipy.integrate as integrate
import pylab
import time
import pandas as pd

xls = pd.ExcelFile('earthquake.xlsx')
df1 = pd.read_excel(xls, 'Sheet1')
nsize=df1.shape[0]
samples=np.empty(shape=(nsize,2)) 
samples[:,0]=df1["Recurrence time"]
samples[:,1]=df1["Magnitude"]


# In[2]:


def exponentiated_quadratic(xa, xb,l):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5* scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')/l/l
    return np.exp(sq_norm)

def chol_sample(mean, cov):
    return mean + np.linalg.cholesky(cov+np.diag(np.full(cov.shape[0], jit))) @ np.random.standard_normal(mean.size)

    
def vec_sum(rho,nu,mu2,X1,sigmasq_x,mu_x):
    s1=rho*math.sqrt(nu)*(X1[:,0]-mu_x)/math.sqrt(sigmasq_x)
    s2=X1[:,1]-mu2-s1
    s3=(s2**2).sum()
    s4=s3/2/(1-rho**2)/nu
    return s4

def log_lik(f,x,ns):
    sig=torch.sigmoid(torch.Tensor(f)).numpy()
    l1=np.log(sig[:,0:ns]).sum()
    l2=np.log(1-sig[:,ns:]).sum()
    return l1+l2

def elliptical_slice(initial_theta,prior,lnpdf,pdf_params=(),
                     cur_lnpdf=None,angle_range=None):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix 
               (like what numpy.linalg.cholesky returns), 
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
    """
    D= len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf= lnpdf(initial_theta,*pdf_params)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1: #prior = prior sample
        nu= prior
    else: #prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu= np.dot(prior,np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi= np.random.uniform()*2.*math.pi
        phi_min= phi-2.*math.pi
        phi_max= phi
    else:
        # Randomly center bracket on current point
        phi_min= -angle_range*np.random.uniform()
        phi_max= phi_min + angle_range
        phi= np.random.uniform()*(phi_max-phi_min)+phi_min

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = initial_theta*math.cos(phi) + nu*math.sin(phi)
        cur_lnpdf = lnpdf(xx_prop,*pdf_params)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform()*(phi_max - phi_min) + phi_min
    return (xx_prop,cur_lnpdf)


def gpr(logtheta,x,y):
    #return the gradient of negative loglikelihood wrt log(lengthscale)
    K=exponentiated_quadratic(x, x,np.exp(logtheta))
    K_inv=np.linalg.inv(K+np.diag(np.full(x.shape[0], jit)))
    
    W=K_inv-K_inv@y.reshape(x.shape[0],1)@y.reshape(x.shape[0],1).T@K_inv
    grad=W*K*(-2*np.log(K))
    return np.ndarray.sum(grad)/2


def logmvnpdf(x, mean, cv_chol):
    #return the loglikelihood
    cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
    cv_sol = scipy.linalg.solve_triangular(cv_chol, (x - mean.reshape(x.shape[0],x.shape[1])), lower=True).T
    n_dim=x.shape[0]
    return - .5 * (np.sum(cv_sol ** 2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det)

def gp_hyp_prior_loglik(logtheta,alpha,beta):
    #without normalizer inverse gamma
    #return the loglikelihood
    return -(alpha)*logtheta-beta*np.exp(-logtheta)

def gp_hyp_prior_grad(logtheta,alpha,beta):
    return -(alpha)+beta*np.exp(-logtheta)

def update_gphp_hmc(logtheta, xx, ff, tau, epsilon,alpha,beta,M):
    K=exponentiated_quadratic(xx, xx,np.exp(logtheta))
    
    chol_K=np.linalg.cholesky(K+np.diag(np.full(xx.shape[0], jit)))
    nlml=-logmvnpdf(ff, np.zeros(ff.shape[0]), chol_K)
    dnlml=gpr(logtheta,xx,ff)


    #% Calculate the prior and gradient.
    logprior=gp_hyp_prior_loglik(logtheta,alpha,beta)
    dlogprior= gp_hyp_prior_grad(logtheta,alpha,beta)                        
    #% Instantiate momenta.#tune this variance
    P = np.random.normal(0,np.sqrt(M),1)

    #% Calculate the Hamiltonian.
    init_H = P**2/2/M + nlml - logprior

    #% Calculate the gradient.
    G = dnlml - dlogprior

    # Initialize
    loghp = logtheta

    # Take tau steps.
    for t in range(tau):
        P = P - epsilon * G/2
        loghp = loghp + epsilon * P/M
        dnlml=gpr(loghp,xx,ff)
        dlogprior=gp_hyp_prior_grad(loghp,alpha,beta)
        G = dnlml - dlogprior
        P = P - epsilon * G/2
    
    K=exponentiated_quadratic(xx, xx,np.exp(loghp))
    chol_K=np.linalg.cholesky(K+np.diag(np.full(xx.shape[0], jit)))
    nlml=-logmvnpdf(ff, np.zeros(ff.shape[0]),chol_K)                         
    logprior=gp_hyp_prior_loglik(loghp,alpha,beta)
    
    new_H = P**2/2/M + nlml - logprior
    
    # Accept or reject.
    u=np.random.uniform(low=0.0, high=1.0, size=1)
    if u<np.exp(init_H - new_H):
        logtheta = loghp

    return logtheta


# In[11]:


nsize=samples.shape[0]

mu_x=np.mean(samples[:,0])
sigmasq_x=np.cov(samples.T)[0,0]
jit=10**(-3)

l=20

d1=0.75
d2=0.15
xrange0=max(math.floor(min(samples[:,0]))-3,0.1)
xrange1=math.ceil(max(samples[:,0]))+3
yrange0=max(math.floor(min(samples[:,1]))-3,0.1)
yrange1=math.ceil(max(samples[:,1]))+3
X = np.arange(xrange0, xrange1,d1)
nn2=len(X)
Y = np.arange(yrange0, yrange1,d2)
nn1=len(Y)

X, Y = np.meshgrid(X, Y)
X,Y=X.flatten(),Y.flatten()
XY=np.column_stack([X,Y])

KXX=exponentiated_quadratic(samples,samples,l)

#the gaussian values at observations
l_x=chol_sample(mean=np.zeros(nsize), cov=KXX)



X1=samples# corresponds to the joint set of observations and virtual rejections

f_accum=l_x.reshape(nsize,1)# the correspoding gaussian values to X1
flag_accum=np.ones(nsize,dtype=bool)
epoch=5000
burnin=1000
ft=np.empty(shape=(epoch-burnin,XY.shape[0])) 

nu_ite=10
rho_ite=-0.8
mu_y_ite=-20
k0=0.001
alpha0=0.001
beta0=0.001
mu0=0
sigmasq_y_vec=np.empty(epoch)
rho_vec=np.empty(epoch)
mu_y_vec=np.empty(epoch)

aa=.001
bb=.001

tau=5
eps=0.002
M=0.01

rem=10
l_vec=np.empty(int(epoch/rem))


alpha_family=0.1
beta_family=0.1
rate_vec=np.empty(epoch)
rate_ite=np.random.gamma(shape=nsize+alpha_family, scale=1/(beta_family+sum(samples[:,0])), size=1)
mean_x_ite=1/rate_ite
var_x_ite=1/(rate_ite**2)



for s in range(epoch):

    ssize=0  
    #initialize flag and sample_x first
    flag=np.ones(nsize,dtype=bool)# will be updated in every while loop;flag and sample_x are paired.
    sample_x=samples[:,0]# will be updated in every while loop; to be X_A at the beginning
   
    while flag.sum()>0:
        ssize=ssize+1
        n=flag.sum()
        sample_x=sample_x[flag].reshape(n,1) #sample_x got updated 
        z=np.random.normal(0,math.sqrt(nu_ite*(1-rho_ite**2)),n).reshape(n,1)
        sample_y=z+mu_y_ite+rho_ite*math.sqrt(nu_ite)/math.sqrt(var_x_ite)*(sample_x-mean_x_ite)
        X2=np.column_stack([sample_x,sample_y])
        K22=exponentiated_quadratic(X2,X2,l)
    
        K11=exponentiated_quadratic(X1,X1,l)
        K12=exponentiated_quadratic(X1,X2,l)
        
        K111=K11+np.diag(np.full(X1.shape[0], jit))
        solved = scipy.linalg.solve(K111, K12, assume_a='pos').T
        K2 = K22 - np.dot(solved , K12)
        M2 = np.dot(solved ,f_accum)
        ff=chol_sample(mean=M2.flatten(), cov=K2)
        f_accum=np.concatenate((f_accum, ff.reshape(n,1)), axis=0)
        X1=np.concatenate((X1, X2), axis=0) 
        acc=torch.sigmoid(torch.Tensor(ff)).numpy()
        u=np.random.uniform(low=0.0, high=1.0, size=n)
        flag=(u>=acc).flatten()
        flag_accum=np.concatenate((flag_accum, flag), axis=0)   
        
    X1=X1[flag_accum,:]
    f_accum=f_accum[flag_accum]
    K11=exponentiated_quadratic(X1,X1,l)
    nn=X1.shape[0]
    flag_accum=np.concatenate((np.ones(nsize,dtype=bool), np.zeros(nn-nsize,dtype=bool)), axis=0)                    
    #elliptical slice sampling
    prior=chol_sample(mean=np.zeros(nn), cov=K11)
    f_accum,curloglike=elliptical_slice(f_accum.reshape(1,-1),prior,log_lik,pdf_params=(X1,nsize),
                     cur_lnpdf=None,angle_range=None)
    f_accum=f_accum.T
    
    
    if s>(burnin-1):
        KXY=exponentiated_quadratic(XY,XY,l)
        K12=exponentiated_quadratic(X1,XY,l)
        K111=K11+np.diag(np.full(X1.shape[0], jit))
        solved = scipy.linalg.solve(K111, K12, assume_a='pos').T
        KY = KXY - np.dot(solved , K12)
        MY = np.dot(solved ,f_accum)
        ft[s-burnin,:]=chol_sample(mean=MY.flatten(), cov=KY)
    
    #update conditional dist \pi_0(x_2|x_1) parameters(rho_ite and nu)
    #fist update rho_ite
    m=nn
    rho_cand=np.random.uniform(low=rho_ite-0.1, high=rho_ite+0.1, size=1)
    if rho_cand>=-1 and rho_cand<=1:
        alpha=((1-rho_ite**2)/(1-rho_cand**2))**(m/2)*np.exp(-vec_sum(rho_cand,nu_ite,mu_y_ite,X1,var_x_ite,mean_x_ite)+vec_sum(rho_ite,nu_ite,mu_y_ite,X1,var_x_ite,mean_x_ite))
        u=np.random.uniform(low=0.0, high=1.0, size=1)
        if u<alpha:
            rho_ite=rho_cand 
           
    #update nu
    epsilon_old=np.log(nu_ite)
    epsilon_cand=np.random.normal(epsilon_old,math.sqrt(0.1),1)
    nu_cand=np.exp(epsilon_cand)
    term1=(k0*(mu_y_ite-mu0)**2+2*beta0)/2/nu_cand
    term2=(k0*(mu_y_ite-mu0)**2+2*beta0)/2/nu_ite
    alpha=(nu_cand/nu_ite)**(-alpha0-(m+1)/2)*np.exp(-term1+term2-vec_sum(rho_ite,nu_cand,mu_y_ite,X1,var_x_ite,mean_x_ite)+vec_sum(rho_ite,nu_ite,mu_y_ite,X1,var_x_ite,mean_x_ite))
    u=np.random.uniform(low=0.0, high=1.0, size=1)
    if u<alpha:
        nu_ite=nu_cand
    
    
    mean_mu_y=k0*mu0*(1-rho_ite**2)-rho_ite*math.sqrt(nu_ite)*((X1[:,0]-mean_x_ite)/math.sqrt(var_x_ite)).sum()+X1[:,1].sum()
    mean_mu_y=mean_mu_y/(m+k0*(1-rho_ite**2))
    
    
    sigmasq_mu_y=k0+m/(1-rho_ite**2)
    sigmasq_mu_y=nu_ite/sigmasq_mu_y
    mu_y_ite=np.random.normal(mean_mu_y,math.sqrt(sigmasq_mu_y),1)
    sigmasq_y_vec[s]=nu_ite
    rho_vec[s]=rho_ite
    mu_y_vec[s]=mu_y_ite
    
    
    #update rate_x
    rate_cand=np.random.gamma(shape=nsize+alpha_family, scale=1/(beta_family+sum(samples[:,0])), size=1)
    mean_x_cand=1/rate_cand
    var_x_cand=1/(rate_cand)**2
    rate_old=rate_ite
    mean_x_old=1/rate_old
    var_x_old=1/(rate_old)**2
    alpha=np.exp(-vec_sum(rho_ite,nu_ite,mu_y_ite,X1,var_x_cand,mean_x_cand)+vec_sum(rho_ite,nu_ite,mu_y_ite,X1,var_x_old,mean_x_old))
    u=np.random.uniform(low=0.0, high=1.0, size=1)
    if u<alpha:
        rate_ite=rate_cand
        mean_x_ite=mean_x_cand
        var_x_ite=var_x_cand
    rate_vec[s]=rate_ite
    
    #update lengthscale parameter
    if s%rem==0:
        l_cand=np.exp(update_gphp_hmc(np.log(l), X1, f_accum, tau, eps,aa,bb,M))
        l_vec[int(s/rem)]=l_cand
        l=l_cand        
    
print('end')





# In[12]:


#ranges of the area
print(xrange0)
print(xrange1)
print(yrange0)
print(yrange1)


#corresponding center location is
print(XY[1799,:])


# In[13]:


#calculate the effective sample size for \lambda
import arviz as az
eff_epp=np.zeros(ft.shape[1])
for j in range(ft.shape[1]):
    eff_epp[j]=az.ess(ft[:,j])

print(min(eff_epp))
print(max(eff_epp))
print(eff_epp[1799])




# In[14]:


#plot traceplots 
plt.figure()
plt.plot(ft[:,1799])
plt.xlabel('latent GP at domain center')


sigmasq_y_vec1=sigmasq_y_vec[burnin:epoch]
rho_vec1=rho_vec[burnin:epoch]
mu_y_vec1=mu_y_vec[burnin:epoch]
rate_vec1=rate_vec[burnin:epoch]

plt.figure()
plt.plot(sigmasq_y_vec1)
plt.xlabel(r"$\sigma^2_2$")

plt.figure()
plt.plot(rho_vec1)
plt.xlabel(r"$\rho$")

plt.figure()
plt.plot(mu_y_vec1)
plt.xlabel(r"$\mu_2$")

plt.figure()
plt.plot(rate_vec1)
plt.xlabel(r"$r$")



# In[15]:


# posterior mean plot

pi_0_vec=np.empty(shape=(epoch-burnin,XY.shape[0])) #(shape=(epoch-burnin,XY.shape[0])) 
pi_0_vec_joint=np.empty(shape=(epoch-burnin,XY.shape[0])) 

for i in range(epoch-burnin):
    meanss=np.array([mu_x,mu_y_vec1[i]])
    covarss=rho_vec1[i]*math.sqrt(sigmasq_x)*math.sqrt(sigmasq_y_vec1[i])
    covss = np.array([[sigmasq_x,covarss], [covarss,sigmasq_y_vec1[i]]])
    pi_0_vec[i,:]=multivariate_normal(meanss, covss).pdf(XY)/norm(loc=mu_x,scale= math.sqrt(sigmasq_x)).pdf(XY[:,0])
    pdf = rate_vec[i]*np.exp(-rate_vec[i]*XY[:,0])
    pi_0_vec_joint[i,:]=pi_0_vec[i,:]*pdf

#test

pdff=pi_0_vec_joint*torch.sigmoid(torch.Tensor(ft)).numpy()
pdff2=pi_0_vec*torch.sigmoid(torch.Tensor(ft)).numpy()

chunk_size = nn1
out = pdff2.reshape(pdff2.shape[0], chunk_size, -1).sum(1)
out2=out*d2 #f(\lambda,x_1)


out3=np.tile(np.array(out2),(1,nn1)) #repeat f(\lambda,x_1) 94 column times
pdff3=pdff/out3
pdf_pos2=np.mean(pdff3, axis=0)


X = np.arange(xrange0, xrange1,d1)
Y = np.arange(yrange0, yrange1,d2)
X, Y = np.meshgrid(X, Y)
X,Y=X.flatten(),Y.flatten()
Xnew=X.reshape(nn1,nn2).T
Ynew=Y.reshape(nn1,nn2).T

pdf_pos_new2=pdf_pos2.reshape(nn1,nn2).T


plt.figure()
plt.xlabel('Recurrence Time')
plt.ylabel('Magnitude')
plt.xlim(Xnew.min(),Xnew.max())
plt.ylim(Ynew.min(),Ynew.max())

plt.contourf(Xnew, Ynew, pdf_pos_new2,cmap='Reds',alpha=1,levels=20)
plt.contour(Xnew, Ynew, pdf_pos_new2,cmap='Reds',linewidths=5)
plt.plot(samples[:,0], samples[:,1],'.',color='black',alpha=0.25)

