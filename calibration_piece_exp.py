# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:20:14 2021

@author: Theodor
"""
"""
The objective is that, given a sequence of bond market prices, to find the lambda 
parameters of the piecewise exponential model. 

"""
import sys
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\credit')
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\bond_price_project\bond_measures')
from piecewise_exponential import piecewise_exponential as piece_expo
from bond_class import bond
#%%
def calibration_piecewise_exp(bonds,r,R,left = True):
    """
    bonds =  objects of class bond. 
    
    I assume a flat interest rate curve: r
    
    R = recovery rate
    Output:
        
    The function returns:
    
    lbds_opt: the optimal intensities, the yield curve, the spreads
    
    
    """
    import numpy as np
    import scipy.integrate as integ
    import scipy.optimize as opt
    """First we compute the yield for each bond"""
    yields = [bonds[i].YTM(mkt_price = bonds[i].mkt_price) for i in range(len(bonds))]
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    unique_maturities = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    if left==True:
        surv = lambda lbds,t:piece_expo(unique_maturities,lbds,left = True).survival(t)
    else:
        surv = lambda lbds,t:piece_expo(unique_maturities[0:-1],lbds).survival(t)
    def integrator(lbds,t1,t2):
        f = lambda t:surv(lbds,t)*np.exp(-r*t)
        return integ.quad(f,t1,t2)[0]
    knots = [0]+unique_maturities
    matrix = lambda lbds:np.array([[integrator(lbds,knots[j],knots[j+1])/\
                        integrator(lbds,knots[0],knots[i]) if j<i else 0 \
                for j in range(0,len(unique_maturities))] for i in range(1,len(knots))])
    avg_yields = [np.mean([yields[i] for (i,x) in enumerate(bonds) if x.T==y]) \
                  for y in unique_maturities]
    spreads = np.array([y-r for y in avg_yields]).T
    f = lambda lbds:np.dot(matrix(lbds),np.array(lbds))-spreads/(1-R)
    lbds_opt = opt.fsolve(f,np.ones((1,len(unique_maturities))))
    if left==True:
        surv2 = lambda t:piece_expo(unique_maturities,lbds_opt,left = True).survival(t)
    else:
        surv2 = lambda t:piece_expo(unique_maturities[0:-1],lbds_opt).survival(t)
    def integrator2(t1,t2):
        f = lambda t:surv2(t)*np.exp(-r*t)
        return integ.quad(f,t1,t2)[0]
    spread_curve = lambda T:np.dot([integrator2(knots[j],knots[j+1])/\
                        integrator2(knots[0],T) if knots[j]<T else 0 \
                for j in range(0,len(unique_maturities))],[lbds_opt[j] if knots[j]<T else 0\
                for j in range(0,len(unique_maturities))]) * (1-R)
    calibrated_spreads = np.array([spread_curve(x) for x in unique_maturities])
    res = sum((spreads-calibrated_spreads)**2)
    return lbds_opt,dict(zip(unique_maturities,avg_yields)),spreads,res

#%%
def calibration_exp(bonds,r,R):
    r"""
    Given the bond objects (bonds), the interest rate r and the recovery rate R
    we find the implied lambda parameter of the exponential model. 
    """
    import numpy as np
    yields = [bonds[i].YTM(mkt_price = bonds[i].mkt_price) for i in range(len(bonds))]
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    unique_maturities = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    avg_yields = [np.mean([yields[i] for (i,x) in enumerate(bonds) if x.T==y]) \
                  for y in unique_maturities]
    spreads = np.array([y-r for y in avg_yields]).T
    lbd_opt = np.mean(spreads)/(1-R)
    res = sum((spreads-(1-lbd_opt)*R)**2)
    return lbd_opt,res
#%%
def calibrated_piece_exp_curve(bonds,r,R):
    """
    Piecewise exponential curves
    """
    import numpy as np 
    import scipy.integrate as integ
    lbds_opt = calibration_piecewise_exp(bonds,r,R)[0]
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    unique_maturities = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    surv = lambda t:piece_expo(unique_maturities,lbds_opt,left = True).survival(t)
    def integrator(t1,t2):
        f = lambda t:surv(t)*np.exp(-r*t)
        return integ.quad(f,t1,t2)[0]
    knots = [0]+unique_maturities
    spread = lambda T:np.dot([integrator(knots[j],knots[j+1])/\
                        integrator(knots[0],T) if knots[j]<T else 0 \
                for j in range(0,len(unique_maturities))],[lbds_opt[j] if knots[j]<T else 0\
                for j in range(0,len(unique_maturities))]) * (1-R)
    return spread

def calibrated_exp_curve(bonds,r,R):
    lbd_opt = calibration_exp(bonds,r,R)[0]
    return lambda t: (1-lbd_opt)*R