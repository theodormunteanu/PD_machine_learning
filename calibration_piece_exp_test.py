# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:07:52 2021

@author: Theodor
"""

import sys
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\credit')
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\bond_price_project\bond_measures')
from bond_yield import bond_yield
from bond_class import bond
from piecewise_exponential import piecewise_exponential as piece_expo
from calibration_piece_exp import calibration_piecewise_exp,calibration_exp
#%%
def yield_bootstrapping():
    dic = {0.5:[(0.02,100.1,4)],1:[(0.03,100.7,2),(0.04,101.3,1)],\
                 2:[(0.03,102.7,1),(0.04,103.1,1)],3:[(0.035,105.1,1)]}
    #expiries = list(dict.keys())
    #print(dict[0.5])
    yields_half = bond_yield(100,0.02,0.5,freq = 4,mkt_price = 100.1)
    yields1 = [bond_yield(100,0.03,1,freq = 2,mkt_price = 100.7),\
               bond_yield(100,0.04,1,freq = 1,mkt_price = 101.3)]
    yields2 = [bond_yield(100,0.03,2,freq = 1,mkt_price = 102.7),\
               bond_yield(100,0.04,2,freq = 1,mkt_price = 103.1)]
    yields3 = bond_yield(100,0.035,3,freq = 1,mkt_price = 105.1)
    print("6M yields",yields_half)
    print("1Y yields",yields1)
    print("2Y yields",yields2)
    print("3Y yields",yields3)
yield_bootstrapping()
#%%
def test_lbd_calibration():
    maturities = [0.5,1,2,3]
    r = 0.01
    surv = lambda lbds,t:piece_expo(maturities,lbds,left = True).survival(t)
    import scipy.integrate as integ
    import numpy as np
    def integrator(lbds,t1,t2):
        f = lambda t:surv(lbds,t)*np.exp(-r*t)
        return integ.quad(f,t1,t2)[0]
    knots = [0]+maturities
    matrix = lambda lbds:np.array([[integrator(lbds,knots[j],knots[j+1])/\
                        integrator(lbds,knots[0],knots[i]) if j<i else 0 \
                for j in range(0,len(maturities))] for i in range(1,len(knots))])
    import scipy.optimize as opt
    yields_half = bond_yield(100,0.02,0.5,freq = 4,mkt_price = 100.1)
    yields1 = [bond_yield(100,0.03,1,freq = 2,mkt_price = 99.7),\
               bond_yield(100,0.04,1,freq = 1,mkt_price = 100.3)]
    yields2 = [bond_yield(100,0.03,2,freq = 1,mkt_price = 102.7),\
               bond_yield(100,0.04,2,freq = 1,mkt_price = 103.1)]
    yields3 = bond_yield(100,0.035,3,freq = 1,mkt_price = 105.1)
    yields = [np.mean(x) for x in [yields_half,yields1,yields2,yields3]]
    spreads = np.array([y-r for y in yields]).T
    print("spreads",spreads)
    f = lambda lbds:np.dot(matrix(lbds),np.array(lbds))-spreads
    lbds_opt = opt.fsolve(f,np.array([1,1,1,1]))
    print("the lambda parameters are ",lbds_opt)
    obj = piece_expo(maturities,lbds_opt,left = True)
    print("the survival probs are ",obj.survivals())
test_lbd_calibration()
#%%
def test_calibration2(r):
    import numpy as np
    import scipy.integrate as integ
    import scipy.optimize as opt
    bonds = [bond(100,0.02,0.5,4,mkt_price = 100.1),bond(100,0.03,1,2,mkt_price = 99.7),\
             bond(100,0.04,1,1,mkt_price = 100.3),bond(100,0.03,2,1,mkt_price = 102.7),\
             bond(100,0.04,2,1,mkt_price = 103.1),bond(100,0.035,3,1,mkt_price = 105.1)]
    yields = [bonds[i].YTM(mkt_price = bonds[i].mkt_price) for i in range(len(bonds))]
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    unique_maturities = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    surv = lambda lbds,t:piece_expo(unique_maturities,lbds,left = True).survival(t)
    def integrator(lbds,t1,t2):
        f = lambda t:surv(lbds,t)*np.exp(-r*t)
        return integ.quad(f,t1,t2)[0]
    knots = [0]+unique_maturities
    matrix = lambda lbds:np.array([[integrator(lbds,knots[j],knots[j+1])/\
                        integrator(lbds,knots[0],knots[i]) if j<i else 0 \
                for j in range(0,len(unique_maturities))] for i in range(1,len(knots))])
    
    avg_yields = [np.mean([yields[i] for (i,x) in enumerate(bonds) if x.T==y]) \
                  for y in unique_maturities]
    print(avg_yields)
    spreads = np.array([y-r for y in avg_yields]).T
    f = lambda lbds:np.dot(matrix(lbds),np.array(lbds))-spreads
    lbds_opt = opt.fsolve(f,np.array([1,1,1,1]))
    print(lbds_opt)
    obj = piece_expo(unique_maturities,lbds_opt,left = True)
    print(obj.survivals())
test_calibration2(0.01)
#%%
def test_calibration_piecewise():
    """
    I have 6 bonds data and I have to find: 
        1. The calibrated piecewise exponential parameters
        
        2. The yield curve on average (the average yield for each expiry)
        
        3.
    """
    from calibration_piece_exp import calibrated_piece_exp_curve
    from calibration_other_distributions import calibration_weibull
    from calibration_other_distributions import calibration_log_logistic
    from calibration_other_distributions import calibration_gompertz
    bonds = [bond(100,0.02,0.5,4,mkt_price = 100.1),bond(100,0.03,1,2,mkt_price = 99.7),\
             bond(100,0.04,1,1,mkt_price = 100.3),bond(100,0.03,2,1,mkt_price = 102.7),\
             bond(100,0.04,2,1,mkt_price = 103.1),bond(100,0.035,3,1,mkt_price = 105.1)]
    r,R = 0.01,0.4
    data = calibration_piecewise_exp(bonds,r,0.4)
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    unique_maturities = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    data1 = calibration_exp(bonds,r,R)
    print("Exponential parameter",data1[0],"and residuals",data1[1])
    """spread = spread curve of calibrated piecewise exponential model"""
    data2 = calibration_piecewise_exp(bonds,r,R,left = False)
    print("Piecewise exponential parameters until {0} years term".format(unique_maturities[-1]),\
          data[0],"Sum of residuals",data[-1])
    print("Piecewise exponential parameters with knots {0}".format([0.5,1,2]),data2[0],"Sum of residuals",data2[-1])
    data3 = calibration_weibull(bonds,r,R)
    print("Weibull parameters",data3)
    data4 = calibration_log_logistic(bonds,r,R)
    print("Log logistic parameters",data4)
    data5 = calibration_gompertz(bonds,r,R)
    print("Gompertz parameters",data5)
test_calibration_piecewise()
#%%

#%%