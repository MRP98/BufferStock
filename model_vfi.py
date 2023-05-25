# Import packages
from types import SimpleNamespace
import numpy as np
import tools

class model_bufferstock():

    def __init__(self):
        """ defines default attributes """

        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
    
    #############
    ### setup ###
    #############

    def setup(self):
        '''Define model parameters'''

        par = self.par

        # Preferences
        par.T = 10          # Terminal age
        par.beta = 0.99     # Discount factor

        # Debt
        par.r_d = 0.02      # Interest
        par.lambdaa = 0.05  # Installment
      
        # Grids
        par.N = 30          # Number of points in grids
        par.w_max = 4.0     # Maximum cash on hand
        par.n_max = 2.0     # Maximum total debt

        # Income parameters
        par.Gamma = 1.02 # Deterministic drift in income
        par.u_prob = 0.07 # Probability of unemployment
        par.low_val = 0.30 # Negative shock if unemployed (Called mu in paper) 
        par.sigma_xi = 0.01*4 # Transitory shock
        par.sigma_psi = 0.00005 # Permanent shock

        ## Shock grid settings
        par.Neps = 8 # number of quadrature points for eps
        par.Npsi = 8 # number of quadrature points for psi


    def create_grids(self):
        
        # Unlock namespaces
        sol = self.sol
        par = self.par




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

        par.w = par.xi_w_vec * par.psi_w_vec # Bruger vi ikke pt.
        assert (1-sum(par.w) < 1e-8), 'the weights does not sum to 1'

    #############
    ### solve ###
    #############

    def solve(self):

        # Unlock namespaces
        sol = self.sol
        par = self.par

        # Solutions
        sol.grid_w = np.zeros((par.T,par.N))
        sol.grid_n = np.zeros((par.T,par.N))
        sol.d = np.zeros((par.T,par.N,par.N))
        sol.c = np.zeros((par.T,par.N,par.N))
        sol.v = np.zeros((par.T,par.N,par.N))    

        # Grids
        grid_n = np.linspace(0,par.n_max,par.N)
        grid_c = np.linspace(0,1,par.N)

        # Maximal new debt, d
        def d_max(n,t):
            if t > 5:
                d_max = 0
            else:
                d_max = max((par.n_max-n),0)
            return d_max

        # Loop over periods, t
        for t in range(par.T-1,-1,-1):                          

            # Loop over state variable, n
            for i_n, n in enumerate(grid_n):

                grid_w = np.linspace(0,par.w_max-n,par.N)
    
                # Loop over state variable, w
                for i_w, w in enumerate(grid_w):
                    
                    sol.grid_w[t,:] = grid_w 
                    sol.grid_n[t,:] = grid_n

                    # Solutions for given new debt, d
                    c_d = np.zeros(par.N)
                    V_d = np.zeros(par.N)

                    # Maximal new debt given n
                    grid_d = np.linspace(0,d_max(n,t),par.N)

                    # Loop over desicion variable, d
                    for i_d, d in enumerate(grid_d):

                        V_next = np.zeros(par.N)
 
                        w_d = w + d         # Starting cash on hand
                        c = w_d * grid_c    # Current consumption     
                        w_d_c = w_d - c     # Ending cash on hand
                        
                        if t<par.T-1:

                            # Value function in next period
                            interest = par.r_d * (n + d)
                            installment = par.lambdaa * (n + d)
                            w_next = w_d_c - installment - interest
                            n_next = n + d + np.zeros(par.N)

                            for s in range(len(par.xi_vec)):
                                
                                weight = par.w[s] # Weight of shock = probability of shock

                                xi = par.xi_vec[s]                  # Size of shock
                                psi = par.psi_vec[s]

                                V_next += weight*tools.interp_2d_vec(sol.grid_n[t+1,:], sol.grid_w[t+1,:], sol.v[t+1,:,:], n_next, w_next*psi + xi)
                
                        # Find solution for given new debt, d
                        V_guess = np.sqrt(c) + par.beta * V_next
                        index = V_guess.argmax()
                        V_d[i_d] = V_next[index]
                        c_d[i_d] = c[index]

                    # Final solution
                    V_guess = np.sqrt(c_d) + par.beta * V_d
                    index = V_guess.argmax()

                    sol.c[t, i_n, i_w] = c_d[index]
                    sol.d[t, i_n, i_w] = grid_d[index]
                    sol.v[t, i_n, i_w] = np.amax(V_guess)  