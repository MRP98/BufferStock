# Import packages
from types import SimpleNamespace
import numpy as np
import tools

class model_bufferstock():

    def __init__(self,name=None):
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
        par.rho = 3 # CRRA parameter
        par.beta = 0.9 # Discount factor

        # Grid points
        par.T = 10           # Terminal age
        par.Na = 200        # Number of points in grid for w
        par.a_max = 5.0     # Maximum point in grid for w
        par.d_max = 1     # Maximum new debt


        # Income parameters
        par.Gamma = 1.02 # Deterministic drift in income
        par.u_prob = 0.07 # Probability of unemployment
        par.low_val = 0.30 # Negative shock if unemployed (Called mu in paper) 
        par.sigma_xi = 0.01*4 # Transitory shock
        par.sigma_psi = 0.00005 # Permanent shock

        ## Shock grid settings
        par.Neps = 8 # number of quadrature points for eps
        par.Npsi = 8 # number of quadrature points for psi

        # 
        par.r_d = 0.03
        par.r_w = 0.02
        par.phi = 1
        par.lambdaa = 0.2


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
        par.Nshocks = par.w.size    # count number of shock nodes

        par.grid_p = np.ones((par.T,par.Na))   

        for t in range(par.T): 
            par.grid_p[t,:] = par.grid_p[t-1,:]*1.02

        # Try to make Gauss Hermite shocks
        # for t in range(par.T): 
        #     if t == 0:
        #         pass
            
        #     else:
        #         for s in range(len(par.psi_vec)):
                
        #             weight = par.w[s] # Weight of shock = probability of shock
                    

        #             par.grid_p[t,:] += weight*(par.grid_p[t-1,:])*par.psi_vec[s]


        par.grid_d = np.ones((par.T,par.Na))   

        for t in range(par.T):
            if t == 0:
                pass

            elif t == par.T-1:
                par.grid_d[t] = par.grid_d[t-1]*0

            else:
                par.grid_d[t] = par.grid_d[t-1]*(1-par.lambdaa)


    #############
    ### solve ###
    #############

    def solve(self,shocks=False):

        # Initialize
        sol = self.sol
        par = self.par

        # Solutions
        sol.grid_w = np.zeros((par.T,par.Na))
        sol.d = np.zeros((par.T,par.Na))
        sol.c = np.zeros((par.T,par.Na))
        sol.v = np.zeros((par.T,par.Na))
             

        # Grids
        grid_w = np.linspace(0,par.a_max,par.Na)
        
        grid_c = np.linspace(0,1,par.Na)

        w_d = np.zeros((par.Na,par.Na))

        # Loop through all but the last period
        for t in range(par.T-1,-1,-1):      
            
            w_max = par.d_max*t+par.a_max
            grid_w = np.linspace(0,w_max,par.Na) 
            sol.grid_w[t,:] = grid_w         


            # Loop over state variable, w
            for i_w, w in enumerate(grid_w):
                

                # Insert permanent income
                                                                           

                w_d[:,:] = w + par.grid_p[t,:]

                for i_d, d in enumerate(par.grid_d[t]):
                    
                    d_max = max((1-par.lambdaa)*sol.d[t, i_d], par.phi*par.grid_p[t,0])

                    w_d[i_d,:] += d_max - par.r_d * sol.d[t, :] + par.r_w * w_d[i_d,:]

                c = w_d * grid_c
                w_d_c = w_d - c       
                V_next = 0      

                if t<par.T-1:
                    
                    for s in range(len(par.xi_vec)):
                        
                        weight = par.w[s] # Weight of shock = probability of shock

                        xi = par.xi_vec[s]                  # Size of shock

                        V_next += weight*np.interp(w_d_c + xi, sol.grid_w[t+1,:], sol.v[t+1,:])
               
                v_guess = (c**(1-par.rho)-1)/(1-par.rho) + par.beta * V_next
                index_ = v_guess.argmax()                        
                index = np.unravel_index(index_, v_guess.shape)

                # Juster grid optimal D
                                

                sol.c[t, i_w] = c[index[0],index[1]]
                sol.d[t, i_w] = par.grid_d[t,:][index[0]]

                sol.v[t, i_w] = np.amax(v_guess)

        return sol