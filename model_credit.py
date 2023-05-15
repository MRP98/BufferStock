# Import packages
from types import SimpleNamespace
import numpy as np
import tools
from scipy.interpolate import griddata

class model_bufferstock():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace()    
    
    #############
    ### setup ###
    #############

    def setup(self):
        '''Define model parameters'''

        par = self.par

        #### 1. Basic parameters ####

        # Demograhpics
        par.T = 160 # Terminal age

        # Preferences
        par.rho = 3 # CRRA parameter
        par.beta = 0.90 # Discount factor

        # Income parameters
        par.Gamma = 1.02 # Age-invariant deterministic drift in income
        
        par.sigma_xi = 0.01*4 # Transitory shock
        par.sigma_psi = 0.01*(4/11) # Permanent shock
        
        par.u_prob = 0.07 # Probability of unemployment
        par.low_val = 0.30 # Negative shock if unemployed (Called mu in paper) 

        # Saving and borrowing
        par.r_a = -1.0148 # Return on savings
        par.r_d_r_a = 1.1236
        par.varpsi = 0.74
        par.eta = 0
        par.lambdaa = 0.03  # Maximum borrowing limit

        # Credit risk
        par.pi_lose = 0.0263
        par.pi_gain = 0.0607
        par.chi_lose = 4

        #### 3. Numerical integration and grids ####

        ## a_grid settings
        par.Na = 10 # number of points in grid for a
        par.a_max = 20 # maximum point in grid for a
        par.a_phi = 1.1 # Spacing in grid 

        ## Shock grid settings
        par.Neps = 8 # number of quadrature points for eps
        par.Npsi = 8 # number of quadrature points for psi


    def create_grids(self):
        
        par = self.par
        
        #### 1. Check parameters ####

        assert (par.rho >= 0), 'not rho > 0'
        assert (par.lambdaa >= 0), 'not lambda > 0'


        #### 2. Shocks ####

        # Define epsilon - Nodes and weights for quadrature
        eps,eps_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Neps)

        # Define psi - Nodes and weights for quadrature
        par.psi,par.psi_w = tools.GaussHermite_lognorm(par.sigma_psi,par.Npsi)

        # Combine discrete (mu) and continuous (epsilon) transitory shocks into one composite shock (xi)
        if par.u_prob > 0:
            par.xi =  np.append(par.low_val+1e-8, (eps-par.u_prob*par.low_val)/(1-par.u_prob), axis=None) # +1e-8 makes it possible to take the log in simulation if low_val = 0
            par.xi_w = np.append(par.u_prob, (1-par.u_prob)*eps_w, axis=None)
            
        
        else: # If no discrete shock then xi=eps
            par.xi = eps
            par.xi_w = eps_w

        ## Vectorize all - Repeat and tile are used to create all combinations of shocks (like a tensor product)
        par.xi_vec = np.tile(par.xi,par.psi.size)       # Repeat entire array x times
        par.psi_vec = np.repeat(par.psi,par.xi.size)    # Repeat each element of the array x times
        par.xi_w_vec = np.tile(par.xi_w,par.psi.size)
        par.psi_w_vec = np.repeat(par.psi_w,par.xi.size)

        # Weights for each combination of shocks
        par.w = par.xi_w_vec * par.psi_w_vec
        assert (1-sum(par.w) < 1e-8), 'the weights does not sum to 1'
        par.Nshocks = par.w.size    # count number of shock nodes
        
        # 3. Insert some form of grid i guess????
        par.grid_c = np.nan + np.zeros([par.T,par.Na+1])
        par.grid_d = np.nan + np.zeros([par.T,par.Na+1])
        par.grid_w = np.nan + np.zeros([par.T,par.Na+1])

        for t in range(par.T):
            par.grid_c[t,:] = np.linspace(0,1,par.Na+1)
            par.grid_d[t,:] = np.linspace(0,par.a_max,par.Na+1)
            par.grid_w[t,:] = np.linspace(0,par.a_max,par.Na+1)               
        
        #### 5. Set seed ####
        
        np.random.seed(2022)



    #############
    ### solve ###
    #############

    def solve(self):

        # Initialize
        sol = self.sol
        par = self.par

        shape=(par.T,par.Na+1)
        sol.c = np.zeros(shape)
        sol.d = np.zeros(shape)
        sol.v = np.zeros((par.T, par.Na+1, 2))
        sol.grid_w = np.zeros(shape)
        
        # Last period, (= consume all) 
        sol.c[par.T-1,:] = np.linspace(0,par.a_max,par.Na+1)
        sol.d[par.T-1,:] = np.linspace(0,par.a_max,par.Na+1)


        # Before last period
        for t in range(par.T-2,-1,-1):
            
            # Solve model with EGM
            for id, d in enumerate(par.grid_d[t,:]):

                sol.grid_w[t,:] = par.grid_w[t,:]

                for iw, w in enumerate(sol.grid_w[t,:]):
                    
                    w_d = w + d

                    c = par.grid_c[t,:]*w_d  

                    d_plus = d
                    
                    w_d_c = w_d - c

                    EV_next = 0

                    for s in range(len(par.xi_vec)):

                        weight = par.xi_w_vec[s] / par.Neps
                        xi = par.xi_vec[s]

                        EV_next += weight*tools.interp_2d(par.grid_w[t,:], par.grid_d[t,:], sol.v[t+1,:], w, d_plus)

                    v_guess = np.sqrt(c) + par.beta * EV_next

                    index = np.unravel_index(v_guess.argmax(), v_guess.shape)
                    if t == 50:
                        print(index)

                    sol.c[t, id, iw] = c[index[0]]

                    sol.v[t, iw] = np.amax(v_guess)

        return sol