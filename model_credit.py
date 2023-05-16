# Import packages
from types import SimpleNamespace
import numpy as np
import tools
from scipy.interpolate import interp2d

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
        par.T = 5 # Terminal age

        # Preferences
        par.rho = 3 # CRRA parameter
        par.beta = 0.9 # Discount factor

        # Income parameters
        par.Gamma = 1.02 # Deterministic drift in income
        par.u_prob = 0.07 # Probability of unemployment
        par.low_val = 0.30 # Negative shock if unemployed (Called mu in paper) 
        par.sigma_xi = 0.01*4 # Transitory shock
        par.sigma_psi = 0.01*(4/11) # Permanent shock

        # Saving and borrowing
        par.r_a = -1.0148 # Return on savings
        par.r_d_r_a = 1.1236
        par.varpsi = 0.74
        par.eta = 0
        par.lambdaa = 0.03 # Maximum borrowing limit

        # Credit risk
        par.pi_lose = 0.0263
        par.pi_gain = 0.0607
        par.chi_lose = 4

        #### 3. Numerical integration and grids ####

        ## a_grid settings
        par.Na = 50 # number of points in grid for a
        par.a_max = 5.0 # maximum point in grid for a
        par.d_max = 0.5 # maximum new debt
        par.n_max = 5.0 # maximum debt

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
        
        #### 3. Initialize grids ####

        par.grid_c = np.nan + np.zeros([par.T,par.Na])          # choice variable
        par.grid_d = np.nan + np.zeros([par.T,par.Na])          # choice variable
        par.grid_w = np.nan + np.zeros([par.T,par.Na])          # state variable
        par.grid_n = np.nan + np.zeros([par.T,par.Na])          # state variable

        for t in range(par.T):
            par.grid_c[t,:] = np.linspace(0,1,par.Na)           # choice variable
            par.grid_d[t,:] = np.linspace(0,par.d_max,par.Na)   # choice variable
            par.grid_w[t,:] = np.linspace(0,par.a_max,par.Na)   # state variable
            par.grid_n[t,:] = np.linspace(0,par.n_max,par.Na)   # state variable               
        
        #### 5. Set seed ####
        
        np.random.seed(2022)

    #############
    ### solve ###
    #############

    def solve(self,shocks=False):

        # Initialize
        sol = self.sol
        par = self.par

        sol.v = np.zeros((par.T,par.Na,par.Na))        
        sol.c = np.zeros((par.T,par.Na,par.Na))
        sol.d = np.zeros((par.T,par.Na,par.Na))
        sol.grid_w = np.zeros((par.T,par.Na))
        sol.grid_n = np.zeros((par.T,par.Na))
        
        # Last period
        for i,w in enumerate(par.grid_w):
            sol.c[par.T-1,i,:] = np.linspace(0,i+par.n_max,par.Na) # Consume all
            sol.d[par.T-1,i,:] = np.linspace(0,par.d_max,par.Na) # Take max debt
            sol.v[par.T-1,i,:] = np.sqrt(sol.c[par.T-1,i,:]) # Value in last period

        # Loop through all but the last period
        for t in range(par.T-2,-1,-1):

            c_share = par.grid_c[t,:]        
            d = par.grid_d[t,:]              

            # Loop over state variable, n
            for i_n, n in enumerate(par.grid_n[t,:]):    

                # Loop over state variable, w
                for i_w, w in enumerate(par.grid_w[t,:]):

                    n_d = n + d       # Debt in next period         
                    w_d = w + n_d     # Assets in current period      
                    c = c_share * w_d # Consumption in current period      
                    w_d_c = w_d - c   # Assets in next period     
                    EV_next = 0       # Initialize EV_next 

                    if shocks == True:

                        # Loop over possible shocks - udgår foreløbigt
                        for s in range(len(par.xi_vec)):

                            weight = par.xi_w_vec[s] / par.Neps # Weight of shock = probability of shock
                            xi = par.xi_vec[s]                  # Size of shock

                            EV_next += weight*np.interp(w_d_c + xi, sol.grid_w[t+1,:], sol.v[t+1,:])
                    
                    else:
                        f_interp = interp2d(sol.grid_w[t+1,:], sol.grid_n[t+1,:], sol.v[t+1,:,:], kind='linear')
                        EV_next = f_interp(w_d_c, n_d)
                    
                    v_guess = np.sqrt(c) + par.beta * EV_next            # Value function
                    flat_index = v_guess.argmax()                        # Find index of highest flat v_guess
                    index = np.unravel_index(flat_index, v_guess.shape)  # Convert flat index to normal matrix index
                    
                    sol.c[t, i_w, i_n] = w_d[index[0]] - w_d_c[index[0]] # Find consumption of that index
                    sol.d[t, i_w, i_n] = n_d[index[1]] - n               # Find debt of that index
                    sol.v[t, i_w, i_n] = np.amax(v_guess)

        return sol