# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:55:06 2020

@author: Theodor
"""

class piecewise_Weibull:
    def __init__(self,maturities,intensities,gamma):
        r"""
        Inputs:
            
        maturities: the knots around which the default intensities change
        
        intensities: the hazard rate of default
        
        gamma: the scale parameters
        
        """
        if len(maturities)==len(intensities)-1:
            self.intensities = intensities
            self.maturities = maturities
            self.gamma = gamma
        else:
            raise TypeError("the number of intensities must be = \
                            with the number of maturities + 1")
    def position(self,t):
        r"""
        We use this function to locate t among the knots (maturities)
        of the piecewise Weibull object. 
        """
        if t<=self.maturities[0]:
            return 0
        elif t>self.maturities[-1]:
            return len(self.intensities)-1
        else:
            for i in range(len(self.maturities)):
                if self.maturities[i]<t and t<=self.maturities[i+1]:
                    return i+1
                
    def survival(self,t):
        r"""
        Same as survival, but we provide directly the analytical formula. 
        It is a faster way of computing the probability of survival. 
        
        """
        import numpy as np
        pos = self.position(t)
        gamma = self.gamma
        if pos==0:
            return np.exp(-self.intensities[0]*t**gamma)
        elif pos==1:
            return np.exp(-self.intensities[0]*self.maturities[0]**gamma-\
                          self.intensities[1]*(t-self.maturities[0])**gamma)
        elif pos == len(self.intensities):
            l1 = self.intensities
            l2 = [self.maturities[i]-self.maturities[i-1] for i in range(1,pos)]
            l2.insert(0,self.maturities[0])
            l2.append(t-self.maturities[-1])
            return np.exp(-np.dot(l1,np.array(l2)**gamma))
        else:
            l1 = [self.intensities[i] for i in range(pos+1)]
            l2 = [self.maturities[i]-self.maturities[i-1] for i in range(1,pos)]
            l2.insert(0,self.maturities[0])
            l2.append(t-self.maturities[pos-1])
            a = np.dot(l1,np.array(l2)**gamma)
            return np.exp(-a)
        
    def intensity(self,t):
        maturities,intensities = self.maturities,self.intensities
        indicators = [(0<= t and t< maturities[0]) if i==0 else ((maturities[i-1]<=t and t<maturities[i]) \
               if 1<=i and i<len(intensities)-1 else (maturities[-1]<t)) for i in range(0,len(intensities))]
        pos = indicators.index(True)
        intens = intensities[pos]
        knot = 0 if pos==0 else sum([maturities[i-1]*(pos==i) \
               for i in range(1,len(intensities))])
        return intens,knot
    
    def pdf(self,t):
        gamma = self.gamma
        intens,knot = self.intensity(t)
        return self.survival(t)*intens*gamma*(t-knot)**(gamma-1)


#%%
"""
To price the convertible debt, we need the volatility of the stock price.


"""
