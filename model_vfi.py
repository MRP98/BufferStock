# Import packages
from types import SimpleNamespace
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
        '''Define model parameters'''

        par = self.par

        # Preferences
        par.T = 5              # Terminal age
        par.beta = 0.90         # Discount factor
        par.rho = 3             # CRRA risk aversion

        # Debt
        par.r_w = 0.02          # Return on assets
        par.r_d = 0.10          # Interest rate
        par.lambdaa = 0.03      # Installment
        par.varphi = 0.74       # Maximal debt
      
        # Grids
        par.N = 20              # Number of points in grids
        par.w_old_max = 4.0     # Maximum assets

        # Income parameters
        par.Gamma = 1.02        # Deterministic drift in income
        par.u_prob = 0.07       # Probability of unemployment
        par.credit_con = 0.10
        par.low_val = 0.3               # Negative shock if unemployed (Called mu in paper) 
        par.sigma_xi = 0.01*(4/11)      # Transitory shock
        par.sigma_psi = 0.01*4          # Permanent shock

        ## Shock grid settings
        par.Neps = 8            # Number of quadrature points for eps
        par.Npsi = 8            # Number of quadrature points for psi

    def allocate(self):
        '''Allocate solutions and auxiliaries'''

        par = self.par
        sol = self.sol
        aux = self.aux
        
        # Solutions
        sol.grid_u = np.zeros((par.T,2))
        sol.grid_n = np.zeros((par.T,par.N))
        sol.grid_w_old = np.zeros((par.T,par.N))
        sol.grid_d_old = np.zeros((par.T,par.N))
        sol.grid_m = np.zeros((par.T,par.N))
        sol.d = np.zeros((par.T,par.N,par.N,2))
        sol.c = np.zeros((par.T,par.N,par.N,2))
        sol.v = np.zeros((par.T,par.N,par.N,2))    

        # Grids
        aux.grid_w_old = np.linspace(1e-8,par.w_old_max,par.N)
        aux.grid_d_old = np.linspace(1e-8,par.varphi,par.N)
        aux.grid_c = np.linspace(1e-8,1,par.N)
        aux.grid_d = np.linspace(1e-8,1,par.N)
        aux.grid_u = np.array([1,0])

        aux.V_guess = np.zeros(par.N)
        aux.V_next = np.zeros(par.N)
        aux.c = np.zeros(par.N)

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

    #############
    ### solve ###
    #############

    def solve_bellman(self,w_old,d_next,t):

        # Unlock namespaces
        sol = self.sol
        par = self.par
        aux = self.aux

        # Import auxiliaries
        remaining_debt = aux.remaining_debt
        installment = aux.installment
        interest = aux.interest
        grid_c = aux.grid_c

        # Initialize V_next
        V_next_unemp = np.zeros(par.N)
        V_next_emp = np.zeros(par.N)
        V_next = np.zeros(par.N)
        
        # Cash on hand, consumption and assets
        m_ = (1 + par.r_w)*w_old - installment - interest + par.Gamma
        m = np.clip(m_, a_min=0, a_max=None) # Kan vi bare ignorere negative værdier?
        
        w = (m + d_next - remaining_debt)/par.Gamma
        
        c = w * grid_c 
        w_c = w - c
        
        # V_next = 0 in terminal period 
        if t<par.T-1:

            # Loop over income shocks
            for s in range(len(par.xi_vec)):
                
                weight = par.w[s]   # Weight of shock = probability of shock
                xi = par.xi_vec[s]  # Size of shock
                psi = par.psi_vec[s]
                m_next = (1+par.r_w)*w_c - d_next*(par.r_d + par.lambdaa)/(par.Gamma*xi) + par.Gamma*xi*psi/(par.Gamma*xi)

                # Expected value next if unemployed
                V_next_unemp += weight*tools.interp_2d_vec(sol.grid_d_old[t+1,:], 
                                                           sol.grid_w_old[t+1,:],
                                                           sol.v[t+1,:,:,1], # u = 1
                                                           d_next, m_next) 
                # Expected value next if employed
                V_next_emp += weight*tools.interp_2d_vec(sol.grid_d_old[t+1,:], 
                                                         sol.grid_w_old[t+1,:],
                                                         sol.v[t+1,:,:,0], # u = 0
                                                         d_next, m_next)
                   
            # Final expected value next as weighted average
            V_next = par.credit_con * V_next_unemp + (1-par.credit_con) * V_next_emp


        # Maximize Bellman equation
        V_guess = c**(1-par.rho)/(1-par.rho) + par.beta * V_next
        index = V_guess.argmax()

        # Export auxiliaries
        aux.V_guess = V_guess
        aux.V_next = V_next
        aux.c = c         
        
        return index 

    def solve(self):

        # Unlock namespaces
        sol = self.sol
        par = self.par
        aux = self.aux

        # Import auxiliaries
        grid_w_old = aux.grid_w_old
        grid_d_old = aux.grid_d_old
        grid_u = aux.grid_u
        grid_d = aux.grid_d

        # Maximal attainable assets in period t
        max_w = lambda t, d_old: par.w_old_max * par.Gamma ** t + max(par.xi_vec) * t + d_old

        # Loop over periods, t
        for t in range(par.T-1,-1,-1):
            
            print("T =========== ", t)                          

            # Loop over state variable, d_old
            for i_d_, d_old in enumerate(grid_d_old):

                grid_w_old = np.linspace(0, par.w_old_max, par.N)
                
                # Loop over state variable, w_old
                for i_w, w_old in enumerate(grid_w_old):

                    # Loop over unemployment dummy, u
                    for u in grid_u:

                        sol.grid_u[t,:] = grid_u

                        sol.grid_w_old[t,:] = grid_w_old 
                        sol.grid_d_old[t,:] = grid_d_old
                        
                        aux.interest = par.r_d * d_old
                        aux.installment = par.lambdaa * d_old
                        aux.remaining_debt = (1-par.lambdaa) * d_old 

                        # Solutions for given debt, d
                        c_given_d = np.zeros(par.N)       # Consumption conditional on optimal debt
                        v_next_given_d = np.zeros(par.N)  # Value conditional on optimal debt                        

                        # If unemployed <=> no credit access
                        if u == 1: 
                            
                            for i_d, d in enumerate(grid_d):
                        
                                # Solve Bellman
                                d_next = d * aux.remaining_debt  * np.ones(par.N)
                                index = self.solve_bellman(w_old,d_next,t)

                                # Temporary solutions
                                c_given_d[i_d] = aux.c[index]
                                v_next_given_d[i_d] = aux.V_next[index]

                            V_guess = c_given_d**(1-par.rho)/(1-par.rho) + par.beta * v_next_given_d
                            index = V_guess.argmax()

                            # Solutions
                            sol.d[t, i_d_, i_w, u] = grid_d[index] * aux.remaining_debt 
                            sol.c[t, i_d_, i_w, u] = c_given_d[index] 
                            sol.v[t, i_d_, i_w, u] = V_guess[index]   

                        # If employed <=> credit access
                        else:

                            # Loop over choice variable, d
                            for i_d, d in enumerate(grid_d):
                                
                                # Solve Bellman
                                d_next = d * par.varphi * np.ones(par.N)

                                index = self.solve_bellman(w_old,d_next,t) 

                                # Temporary solutions
                                c_given_d[i_d] = aux.c[index]
                                v_next_given_d[i_d] = aux.V_next[index]

                            # Final solutions
                            V_guess = c_given_d**(1-par.rho)/(1-par.rho) + par.beta * v_next_given_d
                            index = V_guess.argmax()

                            sol.d[t, i_d_, i_w, u] = grid_d[index] * par.varphi
                            sol.c[t, i_d_, i_w, u] = c_given_d[index]
                            sol.v[t, i_d_, i_w, u] = V_guess[index]