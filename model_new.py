# Import packages
from types import SimpleNamespace
import numpy as np
import tools
from scipy.interpolate import interp2d

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

        par.T = 5           # Terminal age
        par.beta = 0.9      # Discount factor
        par.Na = 500        # Number of points in grid for w
        par.a_max = 5.0     # Maximum point in grid for w
        par.d_max = 0.5     # Maximum new debt

    #############
    ### solve ###
    #############

    def solve(self,shocks=False):

        # Unlock namespaces
        sol = self.sol
        par = self.par

        # Solutions
        sol.grid_w = np.zeros((par.T,par.Na))
        sol.d = np.zeros((par.T,par.Na))
        sol.c = np.zeros((par.T,par.Na))
        sol.v = np.zeros((par.T,par.Na))        

        # Grids
        grid_w = np.linspace(0,par.a_max,par.Na)
        grid_d = np.linspace(0,par.d_max,par.Na)
        grid_c = np.linspace(0,1,par.Na)

        w_d = np.zeros((par.Na,par.Na))

        # Loop through all but the last period
        for t in range(par.T-1,-1,-1):      
            
            w_max = par.d_max*t+par.a_max
            grid_w = np.linspace(0,w_max,par.Na) 
            sol.grid_w[t,:] = grid_w                   

            # Loop over state variable, w
            for i_w, w in enumerate(grid_w):
    
                w_d[:,:] = w
                for i_d, d in enumerate(grid_d):
                    w_d[i_d,:] += d
                c = w_d * grid_c    
                w_d_c = w_d - c       
                EV_next = 0      

                if t<par.T-1:
                    EV_next = np.interp(w_d_c, sol.grid_w[t+1,:], sol.v[t+1,:])
                
                v_guess = np.sqrt(c) + par.beta * EV_next
                index_ = v_guess.argmax()                        
                index = np.unravel_index(index_, v_guess.shape)                                        

                sol.c[t, i_w] = c[index[0],index[1]]
                sol.d[t, i_w] = grid_d[index[1]]
                sol.v[t, i_w] = np.amax(v_guess)

        return sol