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
        par.beta = 0.90     # Discount factor
        par.rho = 3

        # Debt
        par.r_w = 0.02
        par.r_d = 0.10      # Interest
        par.lambdaa = 0.03  # Installment
        par.varphi = 0.74
      
        # Grids
        par.N = 30          # Number of points in grids
        par.w_old_max = 4.0     # Maximum cash on hand
        par.d_old_max = 0.74     # Maximum total debt

        # Income parameters
        par.Gamma = 1.02 # Deterministic drift in income
        par.u_prob = 0.07 # Probability of unemployment
        par.low_val = 0.03 # Negative shock if unemployed (Called mu in paper) 
        par.sigma_xi = 0.02 # Transitory shock
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
        sol.grid_w_old = np.zeros((par.T,par.N))
        sol.grid_d_old = np.zeros((par.T,par.N))
        sol.d = np.zeros((par.T,par.N,par.N))
        sol.c = np.zeros((par.T,par.N,par.N))
        sol.v = np.zeros((par.T,par.N,par.N))    

        # Grids
        grid_d_old = np.linspace(0,par.varphi,par.N)
        grid_c = np.linspace(0.0,1,par.N)

        # # Maximal new debt, d
        def d_max(t):
            if t > 80:
                d_max = 0
            else:
                d_max = par.varphi
            return d_max

        # Loop over periods, t
        for t in range(par.T-1,-1,-1):
            print("T =========== ", t)                          

            # Loop over state variable, n
            for i_n, d_old in enumerate(grid_d_old):

                grid_w_old = np.linspace(0,par.w_old_max,par.N)
    
                # Loop over state variable, w
                for i_w, w_old in enumerate(grid_w_old):
                    
                    sol.grid_w_old[t,:] = grid_w_old 
                    sol.grid_d_old[t,:] = grid_d_old

                    # Solutions for given new debt, d
                    c_given_d = np.zeros(par.N)   #Consumption conditional on optimal debt in period t
                    v_next_given_debt = np.zeros(par.N)   #Value function conditional on optimal debt in period t

                    # Maximal new debt given n
                    grid_d = np.linspace(0,par.varphi,par.N)

                    # Loop over decision variable, d
                    for i_d, d in enumerate(grid_d):

                        V_next = np.zeros(par.N)

                        d_next = d * (0.05*(1-par.lambdaa)*d_old + 0.95*par.varphi) * np.ones(par.N)

                        # Value function in next period
                        interest = par.r_d * d_old
                        installment = par.lambdaa * d_old
                        remaining_debt = (1-par.lambdaa)*d_old 
                        m = np.clip((1 + par.r_w)*w_old - installment - interest - remaining_debt + d_next + par.Gamma, a_min=0, a_max=None)

                        c = m * grid_c 
                      

                        w_c = m - c
                      
                        if t<par.T-1:

                            for s in range(len(par.xi_vec)):
                                
                                weight = par.w[s] # Weight of shock = probability of shock

                                xi = par.xi_vec[s]                  # Size of shock
                                psi = par.psi_vec[s]

                                w_next = (1+par.r_w)*w_c - d_next*(par.r_d + par.lambdaa) + xi

                                V_next += weight*tools.interp_2d_vec(sol.grid_d_old[t+1,:], sol.grid_w_old[t+1,:], sol.v[t+1,:,:], d_next, w_next)    
                
                        # Find solution for given new debt, d
                        V_guess = c**(1-par.rho)/(1-par.rho) + par.beta * V_next
                        index = V_guess.argmax()
                        v_next_given_debt[i_d] = V_next[index]
                        c_given_d[i_d] = c[index]

                    # Final solution
                    V_guess = c_given_d**(1-par.rho)/(1-par.rho) + par.beta * v_next_given_debt
                    index = V_guess.argmax()

                    sol.c[t, i_n, i_w] = c_given_d[index]
                    sol.d[t, i_n, i_w] = grid_d[index]
                    sol.v[t, i_n, i_w] = np.amax(V_guess)  