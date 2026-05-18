import numpy as np
import json, sys, os
from scipy import constants as cst
import mpmath as mp
import scipy.special as sp
from scipy.optimize import root,minimize

class MachineAndBeamParametersGenerator:

    def __init__(self):
        self.intensity_per_bunch   = None 
        self.momentum_compaction   = None 
        self.circumference         = None
        self.voltage               = None
        self.RF_frequency          = None
        self.harmonic_number       = None
        self.Xing                  = None
        self.energy                = None
        self.energy_loss_per_turn  = None
        self.n_ip                  = None       
        self.beta_x                = None
        self.n_bunches             = None
        self.xi_y_max              = None 
        self.a_hg                  = None 
        self.bending_radius        = None
        self.alpha_xy              = None
        self._epsilon_x            = None
        
        self.i4i2                  = 0.0
        self.arc_prop              = 0.8
        self.arc_cell_length       = 2*65
        self.C_q                   = 55/(32*np.sqrt(3))*cst.hbar/(cst.m_e*cst.c)

    def __setattr__(self, name, value):
        try:
            original_value = getattr(self, name) 
            if original_value != value:
                super().__setattr__('_sigma_z',None)
                #print(f"Value {name} has changed from {original_value} to {value}")
        except AttributeError: 
            #print(f"Initializing Value {name} with {value}")
            super().__setattr__('_sigma_z',None)
        super().__setattr__(name, value)
        
    def gamma(self):
        return self.energy*1e3/cst.value('electron mass energy equivalent in MeV')
        
    def f_rev(self):
        return cst.c/self.circumference
    
    def tau_z(self):
        return 2.0*self.energy/self.energy_loss_per_turn/(2+self.i4i2)
    
    def q_s(self):
        cosphis = np.cos(self.phi_s())
        return np.sqrt(self.voltage*2.0*np.pi*self.RF_frequency*self.circumference*np.abs(self.eta())/(self.energy*1E9*cst.c)*cosphis)/(2.0*np.pi)
    
    def natural_spin_tune_spread(self):
        return self.gamma()*cst.value('electron mag. mom. anomaly')*self.sigma_delta_SR()
            
    def eta(self):
        return self.momentum_compaction - 1/self.gamma()**2
        
    def sigma_x(self):
        return np.sqrt(self.epsilon_x()*self.beta_x)

    def sigma_y(self):
        return np.sqrt(self.epsilon_y()*self.beta_y())

    def sigma_delta_SR(self):
        return np.sqrt(self.C_q*self.gamma()**2/self.bending_radius/(2+self.i4i2))

    def piwinski(self):
        return self.sigma_z()*np.tan(self.Xing/2)/self.sigma_x()

    def xi_x(self):
        return self.intensity_per_bunch*cst.value('classical electron radius')/(2*np.pi*self.gamma())*self.beta_x/(self.sigma_x()**2*(1+self.piwinski()**2))

    def xi_y(self):
        return self.intensity_per_bunch*cst.value('classical electron radius')/(2*np.pi*self.gamma())*self.beta_y()/(self.sigma_x()*self.sigma_y()*np.sqrt(1+self.piwinski()**2))
    
    def epsilon_x(self):
        if self._epsilon_x is not None:
            return self._epsilon_x
        else:
            bending_angle_per_cell = self.arc_cell_length/self.bending_radius
            arc_momentum_compaction = self.momentum_compaction/self.arc_prop
            f = np.sqrt(arc_momentum_compaction*(2/bending_angle_per_cell)**2+1/12)*self.arc_cell_length/4
            return self.C_q*self.gamma()**2*(4*f/self.arc_cell_length)**3*(bending_angle_per_cell/2)**3/(1-self.i4i2)
    
    def epsilon_y(self):
        return self.epsilon_x()/self.alpha_xy
    
    def luminosity(self):
        return np.float32(1e-4)*self.intensity_per_bunch**2*self.f_rev()*self.n_bunches/(4.0*np.pi*self.sigma_x()*self.sigma_y()*np.sqrt(1+self.piwinski()**2))

    def interaction_length(self):
        return 2*np.sqrt(self.beta_x*self.epsilon_x())/self.Xing
        
    def error_on_interaction_length(self):
        real_interaction_length = self.sigma_z()/np.sqrt(1+self.piwinski()**2)
        return (self.interaction_length()-real_interaction_length)/real_interaction_length
    
    def beta_y(self):
        return self.interaction_length()*self.a_hg

    def phi_s(self):
        return  np.arcsin((self.energy_loss_per_turn*1E9/self.voltage)%np.pi)
    
    def momentum_acceptance(self):
        return 2*self.q_s()/(self.harmonic_number*self.momentum_compaction)*np.sqrt(1+(self.phi_s()-np.pi/2)*np.tan(self.phi_s()))
    
    def sigma_delta(self):
        sig_d =self.sigma_z()/self.beta_s()
        return sig_d
        
    def sigma_z_SR(self):
        return self.sigma_delta_SR()*self.beta_s()
     
    #################################################################################
    ### Khoi's integral formula
    ### https://gitlab.cern.ch/xbuffat/khoi/-/blob/master/bs_1D_flipflop_hgcw_correction.py?ref_type=heads

    def B(self, x):
        B = self.Xing/x**2
        return(B)

    def D(self, n,a,b):
        D = n/(2*a**2) + 1/(2*b**2)
        return (D)
        
    def G(self, n,a,b,c):
        G = self.Xing**2/(2*a**2)+(2*n)/(n*b**2+c**2)
        return (G)

    def L(self, n,a,b,c,d):
        L = a/np.sqrt(self.Xing**2*(n*b**2+c**2)+4*n*d**2)
        return (L)

    def M(self, n,a,b,c,d):
        M = n/2+2*n*self.L(n,a,b,c,d)**2
        return (M)

    def y_ratio(self, a,b):
        y_ratio = a/b
        return(y_ratio)

    def P(self, n,a,b,c,d,e,f):
        P = 1 - self.y_ratio(e,f)**2/(2*self.M(n,a,b,c,d))
        return (P)

    def Q(self, n,a,b,c):
        Q = (2*np.pi)**(-1/2)*a**(1-n)*((2*cst.value('classical electron radius')*b)/(self.gamma()*c)*np.sqrt(2/np.pi))**n
        return (Q)

    def R(self, n,a,b,c,d,e,f,g):
        R = np.pi*sp.gamma(n/2+1)*self.L(n,a,b,c,d)*self.y_ratio(e,f)*self.Q(n,c,g,a)
        return (R)
    
    def I_0(self, n,a,b,c,d,e,f,g):
        I_0 = self.R(n,a,b,c,d,e,f,g)*self.M(n,a,b,c,d)**(-(n/2+1))*sp.hyp2f1(n/2+1,1/2,1,self.P(n,a,b,c,d,e,f))
        return (I_0)
    
    def I_hg(self,n,a,b,c,d,e,f,g):
        I_hg = self.I_0(n,a,b,c,d,e,f,g)-(self.R(n,a,b,c,d,e,f,g))/(2*self.G(n,a,b,c)*self.beta_y()**2*self.M(n,a,b,c,d)**(n/2+1))*(mp.appellf1(1/2,-1,n/2+1,1,self.y_ratio(e,f)**2-n,self.P(n,a,b,c,d,e,f))+(self.B(d)**2*a**2)/(4*self.G(n,d,b,c)*self.M(n,a,b,c,d))*(n/2+1)*mp.appellf1(1/2,-1,n/2+2,2,self.y_ratio(e,f)**2-n,self.P(n,a,b,c,d,e,f)))
        return I_hg

    def eqm_bunch_lengths_weak_hg_no_cw_asym(self, p):
        A_1 = ((self.momentum_compaction*self.circumference)/(2*np.pi*self.q_s()))**2
        A_2 = 55/(24*np.sqrt(3))*A_1*(self.n_ip*self.tau_z())/4*cst.value('classical electron radius')**2*self.gamma()**5*137
        s_zw_SR = np.sqrt(A_1)*self.sigma_delta_SR()
        s_zs_SR = np.sqrt(A_1)*self.sigma_delta_SR()

        out1 = 2/self.tau_z()*(s_zw_SR**2+A_2*self.I_hg(3,self.sigma_x(),p[0],p[1],self.sigma_x(),self.sigma_y(),self.sigma_y(),self.intensity_per_bunch))-(2/self.tau_z()+self.n_ip*4/3*cst.value('classical electron radius')*self.gamma()**3*self.I_hg(2,self.sigma_x(),p[0],p[1],self.sigma_x(),self.sigma_y(),self.sigma_y(),self.intensity_per_bunch))*p[0]**2

        out2 = 2/self.tau_z()*(s_zs_SR**2+A_2*self.I_hg(3,self.sigma_x(),p[1],p[0],self.sigma_x(),self.sigma_y(),self.sigma_y(),self.intensity_per_bunch))-(2/self.tau_z()+self.n_ip*4/3*cst.value('classical electron radius')*self.gamma()**3*self.I_hg(2,self.sigma_x(),p[1],p[0],self.sigma_x(),self.sigma_y(),self.sigma_y(),self.intensity_per_bunch))*p[1]**2

        return out1, out2

    def eqm_bunch_lengths_weak_hg_no_cw(self, p):
        A_2 = 55/(24*np.sqrt(3))*self.beta_s()**2*(self.n_ip*self.tau_z())/4*cst.value('classical electron radius')**2*self.gamma()**5*137
        s_zw_SR = self.sigma_delta_SR()*self.beta_s()
        out1 = 2/self.tau_z()*(s_zw_SR**2+A_2*self.I_hg(3,self.sigma_x(),p[0],p[0],self.sigma_x(),self.sigma_y(),self.sigma_y(),self.intensity_per_bunch))-(2/self.tau_z()+self.n_ip*4/3*cst.value('classical electron radius')*self.gamma()**3*self.I_hg(2,self.sigma_x(),p[0],p[0],self.sigma_x(),self.sigma_y(),self.sigma_y(),self.intensity_per_bunch))*p[0]**2
        return out1
    #################################################################################

    def sigma_z(self):
        if self._sigma_z == None:
            if self.energy > 50:
                eq_sol_1=root(self.eqm_bunch_lengths_weak_hg_no_cw, [self.sigma_z_SR()*1.5] ,method='krylov' )
            else:
                eq_sol_1=root(self.eqm_bunch_lengths_weak_hg_no_cw, [self.sigma_z_SR()*1.5] )#,method='krylov' )
            sigma_z = eq_sol_1.x[0]
            if eq_sol_1.success == True and sigma_z > self.sigma_z_SR():
                super().__setattr__('_sigma_z',sigma_z)
            else:
                print("No result with given starting guesses")
                super().__setattr__('_sigma_z',self.sigma_z_SR())
        return self._sigma_z

    def tot_SR_power(self):
        return self.n_bunches*self.intensity_per_bunch*self.energy_loss_per_turn*1E9*self.f_rev()*cst.e

    def beta_s(self):
        return (np.abs(self.eta())*self.circumference)/(2*np.pi*self.q_s())

    def adjust_vertical_emittance_to_xi_y_max(self, xi_y_max = 0.1):
        xi_y = self.xi_y()
        if xi_y > xi_y_max:
            self.alpha_xy = self.alpha_xy*(xi_y_max/xi_y)**2

    def adjust_hourglass_to_beta_y(self, beta_y = 0.1):
        self.a_hg *= beta_y/self.beta_y()
        
    def adjust_bunch_intensity_to_max_SR_power(self,max_SR_power=50E6):
        self.intensity_per_bunch = self.intensity_per_bunch*max_SR_power/self.tot_SR_power()

    def adjust_n_bunches_to_max_SR_power(self,max_SR_power=50E6):
        self.n_bunches = self.n_bunches*max_SR_power/self.tot_SR_power()
        
    def adjust_beta_x_to_beta_y(self, beta_y):
        self.beta_x = (self.Xing*beta_y/(2*self.a_hg))**2/self.epsilon_x()
        
    def _penalty_adjust_voltage_for_momentum_acceptance(self,voltage0,momentum_acceptance):
        if voltage0[0] <= self.energy_loss_per_turn*1E9:
            return 10.0
        self.voltage = voltage0[0]
        return np.abs(momentum_acceptance-self.momentum_acceptance())

    def adjust_voltage_for_momentum_acceptance(self,momentum_acceptance=0.02):
        opt_result = minimize(self._penalty_adjust_voltage_for_momentum_acceptance,x0=no[self.energy_loss_per_turn*1.2E9],args=(momentum_acceptance),method='Nelder-mead')

        return opt_result.x[0]
        
    def _penalty_adjust_bunch_intensity_to_xi_y_max(self,intensity0,xi_y_max):
        self.intensity_per_bunch = intensity0[0]
        return np.abs(xi_y_max-self.xi_y())

    def adjust_bunch_intensity_to_xi_y_max(self, xi_y_max = 0.1):
        opt_result = minimize(self._penalty_adjust_bunch_intensity_to_xi_y_max,x0=[1E11],args=(xi_y_max),method='Nelder-mead')
        return opt_result.x[0]

if False:
    param_gen_for_z = MachineAndBeamParametersGenerator()
    param_gen_for_z.intensity_per_bunch =  1.4e11
    param_gen_for_z.n_bunches = 12000
    param_gen_for_z.momentum_compaction = 28.67e-6
    param_gen_for_z.circumference = 90658.728
    param_gen_for_z.voltage = 79E6
    param_gen_for_z.RF_frequency = 400000000.0
    param_gen_for_z.harmonic_number = 121200
    param_gen_for_z.Xing = 0.03
    param_gen_for_z.energy = 45.6
    param_gen_for_z.energy_loss_per_turn = 0.0390
    #param_gen_for_z.sigma_delta_SR = 0.00039
    #param_gen_for_z.tau_z = 1171
    param_gen_for_z.n_ip = 4
    param_gen_for_z.beta_x = 0.09
    param_gen_for_z.a_hg = 1.196
    param_gen_for_z.bending_radius = 10021.0
    param_gen_for_z.arc_cell_length = 104
    param_gen_for_z.alpha_xy = 276
    param_gen_for_z.adjust_hourglass_to_beta_y(beta_y=0.8E-3)
    
    print(f"Machine GHC 2024 September design parameters Z:")
    print(f"  Momentum compaction: {param_gen_for_z.momentum_compaction:.3e}")
    print(f"  Crossing angle: {param_gen_for_z.Xing*1e3:.1f} mrad")
    print(f"  RF voltage: {param_gen_for_z.voltage:.3e}")
    print(f"  xi_x: {param_gen_for_z.xi_x():.4f}")
    print(f"  xi_y: {param_gen_for_z.xi_y():.4f}")
    print(f"  Qs: {param_gen_for_z.q_s():.4f}")
    print(f"  beta s: {param_gen_for_z.beta_s():.4f}")
    print(f"  sigma_z: {param_gen_for_z.sigma_z()*1e3:.1f} mm")
    print(f"  sigma_z_SR: {param_gen_for_z.sigma_z_SR()*1e3:.1f} mm")
    print(f"  sigma_delta: {param_gen_for_z.sigma_delta()*1e2:.3f} %")
    print(f"  sigma_delta_SR: {param_gen_for_z.sigma_delta_SR()*1e2:.3f} %")
    print(f"  tau_z: {param_gen_for_z.tau_z():.1f} turns")
    print(f"  beta_x: {param_gen_for_z.beta_x*1e3:.2f} mm")
    print(f"  beta_y: {param_gen_for_z.beta_y()*1e3:.3f} mm")
    print(f"  emittance x: {param_gen_for_z.epsilon_x()*1e9:.3f} nm")
    print(f"  emittance y: {param_gen_for_z.epsilon_y()*1e12:.3f} pm")
    print(f"  momentum acceptance: {param_gen_for_z.momentum_acceptance()*100:.4f} %")
    print(f"  bunch intensity: {param_gen_for_z.intensity_per_bunch:.2e} particles")
    print(f"  instantaneous luminosity: {param_gen_for_z.luminosity()/1e34:.3f} x 10^34 cm^-2 s^-1")
    print(f"  natural spin tune spread / Qs: {param_gen_for_z.natural_spin_tune_spread()/param_gen_for_z.q_s():.4f}")
    print(f"  interaction_length: {param_gen_for_z.interaction_length()*1e3:.3f} mm (err {param_gen_for_z.error_on_interaction_length()*100:.2f}%)")
    print(f"  beta*y/interaction_length: {param_gen_for_z.a_hg}")
    print(f"  Piwinski: {param_gen_for_z.piwinski()}")
    print(f"  4xi_x/Qs: {4*param_gen_for_z.xi_x()/param_gen_for_z.q_s():.4f}")
    print(f"  Sync. rad power: {param_gen_for_z.tot_SR_power()*1E-6:.1f} MW")
    print(f"  X-Y emittance ratio: {param_gen_for_z.alpha_xy:.3f}")

