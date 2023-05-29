# Import packages
from types import SimpleNamespace
from scipy import optimize
import numpy as np
import tools

class model_bufferstock():

    def __init__(self):
        """ defines default attributes """

        self.par = SimpleNamespace() # Parameter values
        self.sol = SimpleNamespace() # Solution variables
        self.aux = SimpleNamespace() # Auxillary variables
    
    #############
    ### setup ###
    #############

    def setup(self):
        ''' Define model parameters '''

        par = self.par

        # Preferences
        par.T = 10                      # Terminal age
        par.beta = 0.90                 # Discount factor
        par.rho = 0.5                   # CRRA risk aversion

        # Debt
        par.r_a = 0.02                  # Return on assets
        par.r_d = 0.05                  # Interest rate
        par.lambdaa = 0.10              # Installment
        par.varphi = 0.74               # Maximal debt
        par.eta = 0.8
      
        # Grids
        par.N = 100                     # Number of points in grids
        par.n_max = 4.0                 # Maximal net assets
        par.d_max = par.eta * par.n_max # Maximal debt level
        par.n_min = par.d_max * (-1)    # Minimal net assets

        # Income parameters
        par.Gamma = 1.02                # Deterministic drift in income
        par.u_prob = 0.07               # Probability of unemployment
        par.low_val = 0.03              # Negative shock if unemployed (Called mu in paper) 
        par.sigma_xi = 0.02             # Transitory shock
        par.sigma_psi = 0.00005         # Permanent shock

        ## Shock grid settings
        par.Neps = 8                    # Number of quadrature points for eps
        par.Npsi = 8                    # Number of quadrature points for psi

    def allocate(self):
        ''' Allocate solutions and auxiliaries '''

        par = self.par
        sol = self.sol
        aux = self.aux
        
        # Solutions
        sol.w = np.zeros((par.T,par.N,par.N,2)) # Continuation value
        sol.d = np.zeros((par.T,par.N,par.N,2))
        sol.c = np.zeros((par.T,par.N,par.N,2))
        sol.v = np.zeros((par.T,par.N,par.N,2))
        sol.d_keep = np.zeros((par.T,par.N,par.N,2))
        sol.c_keep = np.zeros((par.T,par.N,par.N,2))
        sol.v_keep = np.zeros((par.T,par.N,par.N,2))    

        # Grids
        aux.grid_n = np.linspace(par.n_min,par.n_max,par.N)
        aux.grid_d = np.linspace(1e-8,par.d_max,par.N)
        aux.grid_u = np.array([1,0])

    def create_grids(self):
        
        # Unlock namespaces
        par = self.par

        # Nodes and weights for quadrature
        eps,eps_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Neps)
        par.psi,par.psi_w = tools.GaussHermite_lognorm(par.sigma_psi,par.Npsi)

        # One composite shock (xi)
        if par.u_prob > 0:
            par.xi = np.append(par.low_val+1e-8, (eps-par.u_prob*par.low_val)/(1-par.u_prob), axis=None)
            par.xi_w = np.append(par.u_prob, (1-par.u_prob)*eps_w, axis=None)
        
        # If no discrete shock then xi=eps
        else:
            par.xi = eps
            par.xi_w = eps_w

        # Vectorize all - Repeat and tile are used to create all combinations of shocks (like a tensor product)
        par.xi_vec = np.tile(par.xi,par.psi.size)        # Repeat entire array x times
        par.psi_vec = np.repeat(par.psi,par.xi.size)     # Repeat each element of the array x times
        par.xi_w_vec = np.tile(par.xi_w,par.psi.size)
        par.psi_w_vec = np.repeat(par.psi_w,par.xi.size)

        par.w = par.xi_w_vec * par.psi_w_vec # Bruger vi ikke pt.
        assert (1-sum(par.w) < 1e-8), 'the weights does not sum to 1'

    def utility(self,c):

        return (c**(1-self.par.rho)/(1-self.par.rho))

    #############
    ### solve ###
    #############

    def post_decision(self,t):
        ''' Compute post-decision value function in period t '''

        par = self.par
        sol = self.sol
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        grid_u = aux.grid_u

        for i_u, u_bar in enumerate(grid_u):
            for i_n, n_bar in enumerate(grid_n):
                for i_d, d_bar in enumerate(grid_d):

                    # Post-decision states
                    n_bar_plus = (1 + par.r_a) * n_bar - (par.r_d - par.r_a) * d_bar
                    d_bar_plus = (1 - par.lambdaa) * d_bar

                    # Continuation value
                    w_unemp = tools.interp_2d(grid_n, grid_d, sol.v[t+1,:,:,1], n_bar_plus, d_bar_plus) 
                    w_emp = tools.interp_2d(grid_n, grid_d, sol.v[t+1,:,:,0], n_bar_plus, d_bar_plus)    
                    sol.w[t,i_n,i_d,i_u] = (1 - par.u_prob) * w_emp + par.u_prob * w_unemp

    def solve_last_period(self):
        ''' Solve problem in last period '''

        par = self.par
        sol = self.sol
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        grid_u = aux.grid_u

        # Assume no new debt in last period
        for i_u, u_bar in enumerate(grid_u):
            for i_n, n_bar in enumerate(grid_n):
                for i_d, d_bar in enumerate(grid_d):
                        
                        if n_bar + d_bar <= 0: # Negative net assets => no consumption
                            sol.c[par.T-1,i_n,i_d,i_u] = 0
                            sol.d[par.T-1,i_n,i_d,i_u] = d_bar
                            sol.v[par.T-1,i_n,i_d,i_u] = 0
                            
                        else:
                            sol.c[par.T-1,i_n,i_d,i_u] = n_bar + d_bar # Consume everything
                            sol.d[par.T-1,i_n,i_d,i_u] = d_bar
                            sol.v[par.T-1,i_n,i_d,i_u] = self.utility(n_bar + d_bar)

        print("T =========== ", par.T-1)

    def solve_keeper(self,t,n_bar,d_bar,u):
        ''' Solve keepers's problem in period t given states '''

        sol = self.sol
        par = self.par
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        
        c = self.grid_c(n_bar,d_bar)

        # Post-decision states
        n_bar_plus = (1 + par.r_a) * (n_bar - c) - (par.r_d - par.r_a) * d_bar
        d_bar_plus = (1 - par.lambdaa) * d_bar * np.ones(par.N)
   
        # Interpolation of post-decision value function
        v_plus = tools.interp_2d_vec(grid_n, grid_d, sol.w[t,:,:,u], n_bar_plus, d_bar_plus)

        # Solve Bellman equation
        v = self.utility(c) + par.beta * v_plus
        index = np.argmax(v)
        c_keep = c[index]
        v_keep = v[index]

        return (c_keep, v_keep) 
    
    def solve_adjuster(self,t,n_bar,d_bar,u):
        ''' Solve adjuster's problem in period t given states '''

        sol = self.sol
        par = self.par
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d

        d = self.grid_d(n_bar,d_bar,u)
        n_bar_ = n_bar + d
        d_bar_ = d

        # Interpolation of keeper value function
        v = tools.interp_2d_vec(grid_n, grid_d, sol.v_keep[t,:,:,u], n_bar_, n_bar_)
        
        index = np.argmax(v)
        n_bar_adj = n_bar_[index]
        d_adj = d_bar_[index]
        v_adj = v[index]
        
        c_adj = tools.interp_2d(grid_n, grid_d, sol.c[t,:,:,u], n_bar_adj, d_adj)         
        
        return (c_adj, d_adj, v_adj)

    def grid_d(self,n_bar,d_bar,u):
        ''' Choice set for debt '''

        par = self.par

        x = int(not(u)) 
        lower = max(-n_bar, 0) 
        upper = max(d_bar, x * par.eta * n_bar)
        grid_d = np.linspace(lower, upper, par.N)

        return grid_d
    
    def grid_c(self,n_bar,d):
        ''' Choice set for consumption '''

        par = self.par

        lower = 0
        upper = max(0,n_bar + d)
        grid_c = np.linspace(lower, upper, par.N)

        return grid_c

    def solve_nvfi(self):

        sol = self.sol
        par = self.par
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        grid_u = aux.grid_u

        self.solve_last_period() # Last period

        for t in range(par.T-2,-1,-1):            
            
            print("T =========== ", t)

            # 1. Compute post-decision value
            self.post_decision(t)
 
            # 2. Solve keeper's problem
            for i_n, n_bar in enumerate(grid_n):
                for i_d, d_bar in enumerate(grid_d):
                    for u in grid_u:

                        c_keep, v_keep = self.solve_keeper(t,n_bar,d_bar,u)
                        
                        sol.c_keep[t, i_d, i_n, u] = c_keep
                        sol.v_keep[t, i_d, i_n, u] = v_keep
            
            # 3. Solve adjuster's problem
            for i_n, n_bar in enumerate(grid_n):
                for i_d, d_bar in enumerate(grid_d):
                    for u in grid_u:

                        if u == 1: 
                            sol.c[t, i_d, i_n, u] = sol.c_keep[t, i_d, i_n, u]
                            sol.v[t, i_d, i_n, u] = sol.v_keep[t, i_d, i_n, u]
                            continue # No credit access => no adjuster problem   

                        c_adj, d_adj, v_adj = self.solve_adjuster(t,n_bar,d_bar,u)

                        sol.c[t, i_d, i_n, u] = c_adj
                        sol.d[t, i_d, i_n, u] = d_adj
                        sol.v[t, i_d, i_n, u] = v_adj
