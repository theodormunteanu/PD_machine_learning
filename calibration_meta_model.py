# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:05:59 2021

@author: Theodor
"""
from calibration_piece_exp import calibration_piecewise_exp,calibration_exp,\
     calibrated_piece_exp_curve,calibrated_exp_curve
from other_distributions import weibull_dist,gompertz_dist,log_logistic_dist,\
     calibration_log_logistic,calibration_gompertz,calibration_weibull

class calibration:
    def __init__(self,bonds,r,R):
        self.bonds = bonds
        self.r = r
        self.R = R
    
    def piecewise_exp(self,left = True):
        bonds = self.bonds
        r = self.r
        R = self.R
        return calibration_piecewise_exp(bonds,r,R,left = True)
    
    def exp(self):
        bonds = self.bonds
        r = self.r
        R = self.R
        return calibration_exp(bonds,r,R)
    
    def Weibull(self):
        bonds = self.bonds
        r = self.r
        R = self.R
        return calibration_weibull(bonds,r,R)
    
    def Gompertz(self):
        bonds = self.bonds
        r = self.r
        R = self.R
        return calibration_gompertz(bonds,r,R)
        
    def log_logistic(self):
        bonds = self.bonds
        r = self.r
        R = self.R
        return calibration_log_logistic(bonds,r,R)
    
    def calib_curve_piecewise_exp(self):
        bonds = self.bonds
        r = self.r
        R = self.R
        return calibrated_piece_exp_curve(bonds,r,R)
    
    def calib_curve_exp(self):
        bonds = self.bonds
        r = self.r
        R = self.R
        return calibrated_exp_curve(bonds,r,R)
        