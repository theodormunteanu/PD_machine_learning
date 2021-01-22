# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:25:40 2021

@author: Theodor
"""
import sys
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\bond_price_project\bond_measures')
class weibull_dist:
    def __init__(self,lbd,gamma):
        self.lbd = lbd
        self.gamma = gamma
        
    def spread(self,r,R,T):
        lbd,gamma = self.lbd,self.gamma
        q1 = lbd*T**gamma-r*lbd*1/(gamma+1)*T**(gamma+1)+\
        r**2*lbd*gamma/(2*(gamma+1))*T**(gamma+2) - r*lbd**2/2*T**(2*gamma)+\
        r*lbd**2*gamma*T**(2*gamma+1)/(2*gamma+1)+(lbd**3*T**(3*gamma))/6
        q2 = T-r*T**2/2+r**2*T**3/6-lbd*T**(gamma+1)/(gamma+1)+(lbd*r)/(gamma+2)*T**(gamma+2)+\
        lbd**2*T**(2*gamma+1)/(2*(gamma+1))
        return q1/q2*(1-R)
    

#%%

def calibration_weibull(bonds,r,R):
    r"""
    Given the bond objects (bonds), the interest rate r and the recovery rate R
    we find the implied lambda
    """
    import numpy as np
    import scipy.optimize as opt
    yields = [bonds[i].YTM(mkt_price = bonds[i].mkt_price) for i in range(len(bonds))]
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    expiries = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    avg_yields = [np.mean([yields[i] for (i,x) in enumerate(bonds) if x.T==y]) \
                  for y in expiries]
    spreads = np.array([y-r for y in avg_yields]).T
    spread = lambda lbd,gamma,T:weibull_dist(lbd,gamma).spread(r,R,T)
    h = lambda params:sum([(spread(params[0],params[1],expiries[i])-spreads[i])**2 for i in \
                              range(len(expiries))])
    lbd_opt = opt.least_squares(h,np.array([1.0,1.0]))
    calib_spread_curve = lambda T:weibull_dist(lbd_opt.x[0],lbd_opt.x[1]).spread(r,R,T)
    calib_spreads = np.array([calib_spread_curve(x) for x in expiries])
    res = sum((spreads-calib_spreads)**2)
    return lbd_opt.x,res

#%%
class log_logistic_dist:
    def __init__(self,lbd,gamma):
        self.lbd = lbd
        self.gamma = gamma
    
    def spread(self,r,R,T):
        import scipy.integrate as integ
        import numpy as np
        lbd,gamma = self.lbd,self.gamma
        f = lambda u:np.exp(-r*u**gamma)
        g = lambda u:np.exp(-r*u**gamma)*(gamma*u**(gamma-1))/(1+gamma*u)
        q1 = (1-R)*lbd*integ.quad(f,0,T**(1/gamma))[0]
        q2 = integ.quad(g,0,T**(1/gamma))[0]
        return q1/q2

def calibration_log_logistic(bonds,r,R):
    import numpy as np
    import scipy.optimize as opt
    yields = [bonds[i].YTM(mkt_price = bonds[i].mkt_price) for i in range(len(bonds))]
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    expiries = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    avg_yields = [np.mean([yields[i] for (i,x) in enumerate(bonds) if x.T==y]) \
                  for y in expiries]
    spreads = np.array([y-r for y in avg_yields]).T
    spread = lambda lbd,gamma,T:log_logistic_dist(lbd,gamma).spread(r,R,T)
    h = lambda params:sum([(spread(params[0],params[1],expiries[i])-spreads[i])**2 for i in \
                              range(len(expiries))])
    lbd_opt = opt.least_squares(h,np.array([1.0,1.0]))
    calib_spread_curve = lambda T:log_logistic_dist(lbd_opt.x[0],lbd_opt.x[1]).spread(r,R,T)
    calib_spreads = np.array([calib_spread_curve(x) for x in expiries])
    res = sum((spreads-calib_spreads)**2)
    return lbd_opt.x,res
#%%
class gompertz_dist:
    def __init__(self,lbd,gamma):
        self.lbd = lbd
        self.gamma = gamma
    
    def spread(self,r,R,T):
        import scipy.integrate as integ
        import numpy as np
        lbd,gamma = self.lbd,self.gamma
        f = lambda u:u**(-r/gamma)*np.exp(-lbd*u)
        g = lambda u:u**(-r/gamma-1)*np.exp(-lbd*u)
        q1 = (1-R)*lbd*gamma*integ.quad(f,1,np.exp(gamma*T))[0]
        q2 = integ.quad(g,1,np.exp(gamma**T))[0]
        return q1/q2

def calibration_gompertz(bonds,r,R):
    import numpy as np
    import scipy.optimize as opt
    yields = [bonds[i].YTM(mkt_price = bonds[i].mkt_price) for i in range(len(bonds))]
    sorted_bonds = sorted(bonds,key = lambda x:x.T)
    expiries = list(set(sorted_bonds[i].T for i in range(len(bonds))))
    avg_yields = [np.mean([yields[i] for (i,x) in enumerate(bonds) if x.T==y]) \
                  for y in expiries]
    spreads = np.array([y-r for y in avg_yields]).T
    spread = lambda lbd,gamma,T:gompertz_dist(lbd,gamma).spread(r,R,T)
    h = lambda params:sum([(spread(params[0],params[1],expiries[i])-spreads[i])**2 for i in \
                              range(len(expiries))])
    lbd_opt = opt.least_squares(h,np.array([1.0,1.0]))
    calib_spread_curve = lambda T:gompertz_dist(lbd_opt.x[0],lbd_opt.x[1]).spread(r,R,T)
    calib_spreads = np.array([calib_spread_curve(x) for x in expiries])
    res = sum((spreads-calib_spreads)**2)
    return lbd_opt.x,res
"""
def test_gompertz_weibull():
    lbd,gamma,r,R,T = 0.2,0.5,0.02,0.4,1
    spread = lambda t:gompertz_dist(lbd,gamma).spread(r,R,t)
    spread2 = lambda t:log_logistic_dist(lbd,gamma).spread(r,R,t)
    print(spread(0.5),spread(1))
    print(spread2(0.5),spread2(1))
test_gompertz_weibull()"""
#%%