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
                            n_next = n + d

                            V_next = tools.interp_2d_vec(sol.grid_w[t+1,:], sol.grid_n[t+1,:], sol.v[t+1,:,:], w_next, n_next)
                
                        # Find solution for given new debt, d
                        V_guess = np.sqrt(c) + par.beta * V_next
                        index = V_guess.argmax()
                        V_d[i_d] = V_next[index]
                        c_d[i_d] = c[index]

                    # Final solution
                    V_guess = np.sqrt(c_d) + par.beta * V_d
                    index = V_guess.argmax()

                    sol.c[t,i_w,i_n] = c_d[index]
                    sol.d[t,i_w,i_n] = grid_d[index]
                    sol.v[t,i_w,i_n] = np.amax(V_guess)  