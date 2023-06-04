# Import packages
from types import SimpleNamespace
from scipy import optimize
import numpy as np
import tools
import matplotlib.pyplot as plt

class model_bufferstock():

    def __init__(self):
        """ defines default attributes """

        self.par = SimpleNamespace() # Parameter values
        self.sol = SimpleNamespace() # Solution variables
        self.aux = SimpleNamespace() # Auxillary variables
    
    #############
    ### Setup ###
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
        par.r_d = 0.10                  # Interest rate
        par.lambdaa = 0.03              # Installment
        par.varphi = 0.74               # Credit constraint - income
        par.eta = 0.8                   # Credit constraint - net assets
      
        # Grids
        par.N = 40                                          # Number of points in grids
        par.n_max = 4.0                                     # Maximal initial net assets
        par.n_min = par.eta * par.n_max * (-1) - par.varphi # Minimal initial net assets
        par.d_max = par.eta * par.n_max + par.varphi        # Maximal initial debt level

        # Income parameters
        par.Gamma = 1.02                # Deterministic drift in income
        par.u_prob = 0.07               # Probability of being unemployment
        par.credit_con = 0.10           # Probaility of being credit constrained
        par.low_val = 0.3               # Negative shock if unemployed
        par.sigma_xi = 0.01/(4/11)      # Transitory shock
        par.sigma_psi = 0.01*4          # Permanent shock

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
        aux.grid_u = np.array([0,1])

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
        ''' Choice set for debt '''

        par = self.par

        if grid_points is None:
            x = int(not(u)) 
            lower = max(-n_bar, 0) 
            upper = max(d_bar, x * par.eta * n_bar)
            grid_d = np.linspace(lower, upper, par.N)
        else:
            x = int(not(u)) 
            lower = max(-n_bar, 0) 
            upper = max(d_bar, x * par.eta * n_bar)
            grid_d = np.linspace(lower, upper, grid_points) # Used for state spaces

        return grid_d
    
    def grid_c(self,n_bar,d,grid_points=None):
        ''' Choice set for consumption '''

        par = self.par

        if grid_points is None:
            lower = 0
            upper = max(0,n_bar + d)
            grid_c = np.linspace(lower, upper, par.N)
        else:
            lower = 0
            upper = max(0,n_bar + d)
            grid_c = np.linspace(lower, upper, grid_points) # Used for state spaces

        return grid_c
    
    def grid_d(self,n_bar,d_bar,u,grid_points=None):
        ''' Choice set for debt '''

        par = self.par

        if grid_points is None:
            x = int(not(u)) 
            lower = 0 
            upper = max(d_bar, x * (par.eta * n_bar + par.varphi))
            grid_d = np.linspace(lower, upper, par.N)
        else:
            x = int(not(u)) 
            lower = 0
            upper = max(d_bar, x * (par.eta * n_bar + par.varphi))
            grid_d = np.linspace(lower, upper, grid_points) # Used for state spaces

        return grid_d    
    
    #############
    ### Solve ###
    #############

    def post_decision(self,t):
        ''' Compute post-decision value function in period t '''

        par = self.par
        sol = self.sol
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        grid_u = aux.grid_u

        for u in grid_u:
            for i_d, d in enumerate(grid_d):
                for i_n, n in enumerate(grid_n):
                    
                    w_unemp = 0
                    w_emp = 0

                    for s in range(len(par.xi_vec)):
                        
                        # Shocks
                        weight = par.w[s]
                        xi = par.xi_vec[s]
                        psi = par.psi_vec[s]

                        n_bar_plus = ((1 + par.r_a) * n - (par.r_d - par.r_a) * d)/(par.Gamma*psi)
                        d_bar_plus = ((1 - par.lambdaa) * d)/(par.Gamma*psi)

                        # Continuation value (unemployed and employed)
                        w_unemp += weight * tools.interp_2d(grid_d, grid_n, sol.v[t+1,:,:,1], d_bar_plus, n_bar_plus+xi) 
                        w_emp += weight * tools.interp_2d(grid_d, grid_n, sol.v[t+1,:,:,0], d_bar_plus, n_bar_plus+xi)    
                    
                    sol.w[t,i_d,i_n,u] = (1 - par.credit_con) * w_emp + par.credit_con * w_unemp

    def solve_last_period(self):
        ''' Solve problem in last period '''

        par = self.par
        sol = self.sol
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        grid_u = aux.grid_u

        # Assuming no new debt in last period
        for u in grid_u:
            for i_d, d_bar in enumerate(grid_d):
                for i_n, n_bar in enumerate(grid_n):
                        
                        if n_bar + d_bar <= 0: # Negative net assets => no consumption
                            sol.c[par.T-1,i_d,i_n,u] = 0
                            sol.d[par.T-1,i_d,i_n,u] = d_bar
                            sol.v[par.T-1,i_d,i_n,u] = 0
                            
                        else: # Consume everything
                            sol.c[par.T-1,i_d,i_n,u] = n_bar + d_bar
                            sol.d[par.T-1,i_d,i_n,u] = d_bar
                            sol.v[par.T-1,i_d,i_n,u] = self.utility(n_bar + d_bar)

        sol.c_keep[par.T-1,:,:,:] = sol.c[par.T-1,:,:,:]
        sol.d_keep[par.T-1,:,:,:] = sol.d[par.T-1,:,:,:]
        sol.v_keep[par.T-1,:,:,:] = sol.v[par.T-1,:,:,:]

        print("T =========== ", par.T-1)

    def solve_keeper(self,t,n_bar,d_bar,u):
        ''' Solve keepers's problem in period t given states '''

        sol = self.sol
        par = self.par
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        
        d = d_bar # Assuming fixed debt choice
        c = self.grid_c(n_bar,d)
        n = n_bar - c
   
        # Interpolation of post-decision value function
        v_plus = tools.interp_2d_vec(grid_d, grid_n, sol.w[t,:,:,u], d * np.ones(par.N), n)

        # Solve Bellman equation
        v = self.utility(c) + par.beta * v_plus
        index = np.argmax(v)
        c_keep = c[index]
        v_keep = v[index]

        return (c_keep, v_keep) 
    
    def solve_adjuster(self,t,n_bar,d_bar,u,i_n):
        ''' Solve adjuster's problem in period t given states '''

        sol = self.sol
        aux = self.aux

        grid_d = aux.grid_d

        d = self.grid_d(n_bar,d_bar,u)

        # Interpolation of keeper's value function
        v = tools.interp_linear_1d(grid_d, sol.v_keep[t,:,i_n,u], d)
        
        index = np.argmax(v)
        d_adj = d[index]
        v_adj = v[index]

        # Interpolation of keeper's consumption function
        c_adj = tools.interp_linear_1d_scalar(grid_d, sol.c_keep[t,:,i_n,u], d_adj)         
        
        return (c_adj, d_adj, v_adj)

    def solve_nvfi(self,keeper_only=False):
        ''' Solve model with nested value function iterations '''

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
            for u in grid_u:
                for i_d, d_bar in enumerate(grid_d):
                    for i_n, n_bar in enumerate(grid_n):

                        c_keep, v_keep = self.solve_keeper(t,n_bar,d_bar,u)
                        
                        sol.c_keep[t, i_d, i_n, u] = c_keep
                        sol.d_keep[t, i_d, i_n, u] = d_bar
                        sol.v_keep[t, i_d, i_n, u] = v_keep

                        if keeper_only == True:
                            sol.c[t, i_d, i_n, u] = c_keep
                            sol.d[t, i_d, i_n, u] = d_bar
                            sol.v[t, i_d, i_n, u] = v_keep
            
            if keeper_only == False:

                # 3. Solve adjuster's problem
                for u in grid_u:
                    for i_d, d_bar in enumerate(grid_d):
                        for i_n, n_bar in enumerate(grid_n):

                            if u == 1: # No credit access => no adjuster problem
                                sol.c[t, i_d, i_n, u] = sol.c_keep[t, i_d, i_n, u]
                                sol.d[t, i_d, i_n, u] = sol.d_keep[t, i_d, i_n, u]
                                sol.v[t, i_d, i_n, u] = sol.v_keep[t, i_d, i_n, u]
                                    
                            else:
                                c_adj, d_adj, v_adj = self.solve_adjuster(t,n_bar,d_bar,u,i_n)

                                sol.c[t, i_d, i_n, u] = c_adj
                                sol.d[t, i_d, i_n, u] = d_adj
                                sol.v[t, i_d, i_n, u] = v_adj

    def solve_vfi(self):
        ''' Solve model with value function iterations '''
        
        par = self.par
        sol = self.sol
        aux = self.aux

        grid_n = aux.grid_n
        grid_d = aux.grid_d
        grid_u = aux.grid_u

        for t in range(par.T-1,-1,-1):   
            
            print('T = ',t)
            
            for u in grid_u:
                for i_d, d_bar in enumerate(grid_d):
                    for i_n, n_bar in enumerate(grid_n):
                        
                        v = -np.inf
                        d_grid = self.grid_d(n_bar,d_bar,u)
                        
                        for d in d_grid:
                            
                            c_grid = self.grid_c(n_bar,d)

                            for c in c_grid:

                                w_unemp = 0
                                w_emp = 0
                                v_plus = 0
                                
                                if t < par.T-1:

                                    for s in range(len(par.xi_vec)):

                                        # Shocks
                                        weight = par.w[s]
                                        xi = par.xi_vec[s]
                                        psi = par.psi_vec[s]

                                        n_bar_plus = ((1 + par.r_a) * (n_bar - c) - (par.r_d - par.r_a) * d)/(par.Gamma*psi)
                                        d_bar_plus = ((1 - par.lambdaa) * d)/(par.Gamma*psi)

                                        # Continuation value (unemployed and employed)
                                        w_unemp += weight * tools.interp_2d(grid_d, grid_n, sol.v[t+1,:,:,1], d_bar_plus, n_bar_plus+xi) 
                                        w_emp += weight * tools.interp_2d(grid_d, grid_n, sol.v[t+1,:,:,0], d_bar_plus, n_bar_plus+xi)
                                    
                                    v_plus = (1-par.credit_con) * w_unemp + par.credit_con * w_emp

                                v_guess = self.utility(c) + par.beta * v_plus
                                
                                if v_guess > v:
                                    v = v_guess
                                    sol.v[t,i_d,i_n,u] = v
                                    sol.c[t,i_d,i_n,u] = c
                                    sol.d[t,i_d,i_n,u] = d
            
    #################
    ### Graveyard ###
    #################

    def state_space(self,approx_points):
        ''' Compute state space '''

        par = self.par
        aux = self.aux

        # Search grids
        search_grid_d = np.linspace(0.0,3.5,approx_points) # Narrow interval for faster approximation
        search_grid_n = np.linspace(-3.5,4.5,approx_points)

        # Initial state space at t = 0
        grid_n_bar = np.linspace(par.n_min,par.n_max,30)
        grid_d_bar = np.linspace(1e-8,par.d_max,30)
        d, n = np.meshgrid(grid_d_bar,grid_n_bar)
        grid_u = aux.grid_u

        # Container for state spaces
        aux.state_spaces_approx = [(d,n)]
        aux.state_spaces_true = [(d,n)]

        for t in range(1):

            n_bar_plus_ = []
            d_bar_plus_ = []

            print("T =========== ", t+1)

            for u in grid_u:
                for d_bar in grid_d_bar:
                    for n_bar in grid_n_bar:
                        
                        grid_d = self.grid_d(n_bar, d_bar, u, grid_points=35) # Choice set for d

                        for d in grid_d:
                            
                            grid_c = self.grid_c(n_bar, d, grid_points=35) # Choice set for c

                            for c in grid_c:

                                d_bar_plus = (1 - par.lambdaa) * d 
                                n_bar_plus = (1 + par.r_a) * (n_bar - c) - (par.r_d - par.r_a) * d
                    
                                d_bar_plus_.append(d_bar_plus)
                                n_bar_plus_.append(n_bar_plus)

            # True state space
            d_bar_plus_ = np.array(d_bar_plus_)
            n_bar_plus_ = np.array(n_bar_plus_)

            aux.state_spaces_true.append((d_bar_plus_,n_bar_plus_))

            # Approximated state space
            gd, gn = self.approximate_ss(search_grid_d,search_grid_n,d_bar_plus_,n_bar_plus_,threshold=0.075)

            aux.state_spaces_approx.append((gd, gn))

            grid_d_bar = gd[1,:]
            grid_n_bar = gn[:,1]

    def plot_state_space(self,t_plot=None,true_space=False):
        ''' Plot the state space '''
            
        aux = self.aux

        if t_plot != None:
            
            if true_space == True:
                
                print("Warning: Takes a while to plot. Don't lose patience ... ")

                plt.scatter(aux.state_spaces_true[t_plot][0],
                            aux.state_spaces_true[t_plot][1],
                            s=1,label='True state space at $t =$ '+str(t_plot))
            
            plt.scatter(aux.state_spaces_approx[t_plot][0],
                        aux.state_spaces_approx[t_plot][1],
                        s=2,color='red',label='Approximation')
            
        if t_plot == None:
            
            for t in range(len(aux.state_spaces_approx)):
                plt.scatter(aux.state_spaces_approx[t][0],
                            aux.state_spaces_approx[t][1],
                            s=2,label='Approximation at $t =$ '+str(t))

        plt.ylabel(r'$\bar{n}$')
        plt.xlabel(r'$\bar{d}$')
        plt.legend()  

    def approximate_ss(self,search_gridx,search_gridy,state_gridx,state_gridy,threshold=0.05):
        ''' Approximate state space in a given period '''

        # Search grid
        gridX, gridY = np.meshgrid(search_gridx, search_gridy)
        
        # Threshold value for distance
        threshold = threshold
        
        # Grid to be approximated (state grid)
        gridx = state_gridx
        gridy = state_gridy 

        mask = False * np.empty_like(gridX, dtype=bool)

        for (x,y) in  zip(gridx, gridy):
            
            pX = x * np.ones_like(gridX)
            pY = y * np.ones_like(gridY)
            
            distX = (pX - gridX)**2
            distY = (pY - gridY)**2
            
            dist = np.sqrt(distX + distY)
            
            condition = (dist < threshold)
            
            mask = mask | condition

        # Approximated grids
        gX = gridX*mask
        gY = gridY*mask

        return (gX, gY)

    def clean_ss(self):
        ''' Convert from mesh grid to 1D arrays and convert 0 to 1e-8 '''

        aux = self.aux

        for i in range(len(aux.state_spaces_approx)):
            
            # Convert from tuple (immutable) to list (mutable)
            aux.state_spaces_approx[i] = list(aux.state_spaces_approx[i])

            # Convert to 1D arrays
            d_bar = aux.state_spaces_approx[i][0].flatten()
            n_bar = aux.state_spaces_approx[i][1].flatten()

            # Convert 0 to 1e-8 (for interpolation)
            d_zero_ind = np.where(d_bar == 0)[0]
            n_zero_ind = np.where(n_bar == 0)[0]

            d_bar[d_zero_ind] = 1e-8
            n_bar[n_zero_ind] = 1e-8

            aux.state_spaces_approx[i][0] = d_bar
            aux.state_spaces_approx[i][1] = n_bar