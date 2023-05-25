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

        # Debt
        par.r_d = 0.03      # Interest
        par.lambdaa = 0.05  # Installment
      
        # Grids
        par.N = 10          # Number of points in grids
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
            if t > 140:
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

                    # Maximal new debt given n
                    grid_d = np.linspace(0,d_max(n,t),par.N)

                    V_next = np.zeros(par.N)

                    w_d = w + grid_d         # Starting cash on hand
                    c = w_d * grid_c    # Current consumption     
                    w_d_c = w_d - c     # Ending cash on hand
                    
                    if t<par.T-1:

                        # Value function in next period
                        interest = par.r_d * (n + grid_d)
                        installment = par.lambdaa * (n + grid_d)
                        w_next = w_d_c - installment - interest
                        n_next = n + grid_d

                        V_next = tools.interp_2d_vec(sol.grid_n[t+1,:], sol.grid_w[t+1,:], sol.v[t+1,:,:], n_next, w_next)
            
                    # Find solution for given new debt, d
                    V_guess = np.sqrt(c) + par.beta * V_next

                    index = V_guess.argmax()

                    sol.c[t,i_n,i_w] = c[index]
                    sol.d[t,i_n,i_w] = grid_d[index]
                    sol.v[t,i_n,i_w] = np.amax(V_guess)  