import numpy as np
import scipy.linalg

#tester les dérivées numériques
def deriv_num(J,a,d,compute_grad=True,compute_Hess=True) :
    """test numerically the derivative and the Hessian of a function.
        
    Parameters
    ----------
    J : instance of a class
        The function to be tested it must have the following methods, where x is a 1d vector
        of size n
            -- J.eval(x) : evaluation of J at point x, must return a float
            -- J.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- J.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    a : 1d vector of size n
        Point at which the numerical derivatives are evaluated
    d : 1d vector of size n
        Direction in which the numerical derivatives are evaluated
    compute_grad : Boolean
        Flag that tests the function J.grad against numerical derivatives
    compute_Hess : Boolean
        Flag that tests the function J.Hess against numerical derivatives of J.grad
    
   Ouput 
   -----
   This function does not have an output, it prints a string s.
    """
        
    eps_range=[0.1**(i+1) for i in range(12)]
    for eps in  eps_range:
        s='eps {:1.3e}'.format(eps)
        if compute_grad :
            # ratio of numerical derivatives of J and prediction given by J.grad
            ratio=(J.value(a+eps*d)-J.value(a))/(eps*np.dot(J.grad(a),d)) 
            s+=' grad {:1.1e}'.format(np.abs(ratio-1)) 
        if compute_Hess :
            v1=(J.grad(a+eps*d)-J.grad(a))/eps # numerical derivative of J.grad
            v2=J.Hess(a).dot(d)  # prediction given by J.Hess
            ratio=np.linalg.norm(v1)/np.linalg.norm(v2) #norm ratio
            angle=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) # cosinus of the angle between the vectors
            s+=' ratio {:1.1e}'.format(np.abs(ratio-1.))
            s+=' angle {:1.1e}'.format(angle-1.)
        print(s)

        
class BFGS() :
    def __init__(self,nb_stock_max=8) :
       self.stock = []
       self.last_iter = []
       self.nb_stock_max = nb_stock_max
    def push(self, x, grad) :
        if len(self.last_iter) !=0 : 
            x1, gf1 = self.last_iter
            sigma = x-x1
            y = grad-gf1
            rho=1/(sigma.T @ y)
            if rho >= 0 :
                if len(self.stock) >= self.nb_stock_max :
                    self.stock.pop(0)
                self.stock.append([sigma, y, rho])
            else :
                print("Error!")
                self.stock = []
        self.last_iter = [x, grad]
        
    def get(self, grad):
            r = -grad
            alpha=[]
            if len(self.stock) != 0 : 
                for i in range(0, len(self.stock)) :
                    sigmai, yi, rhoi = self.stock[len(self.stock)-i-1]
                    alpha.append(rhoi*(sigmai @ r))
                    r -= alpha[-1] * yi
                r = (self.stock[0][0].T @ self.stock[0][1])/(self.stock[0][1].T @ self.stock[0][1])*r
                for i in range(0, len(self.stock)) :
                    sigmai, yi, rhoi = self.stock[i]
                    beta = rhoi * (yi.T @ r)
                    r+= (alpha[len(self.stock)-i-1]-beta) * sigmai
            return r
        
    def dc(self,x,function,df) :
            self.push(x, df)
            return self.get(df), None




#Algo principal      
def main_algorithm(function,step,xini,dc,ls,itermax = 20000,tol=1.e-4,verbose=True):
    """Perform a minimization algorithm of a function
    
    Parameters
    ----------
    function : instance of a class
        The function to be minimized, depending on the choice of linesearch and direction of descent,
        it must have the following methods, where x is a 1d vector of size n
            -- function.eval(x) : evaluation of J at point x, must return a float
            -- function.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- function.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    step : positive float
        Initial guess of the step
    xini : 1d vector of size n
        initial starting point
    dc : callable
        descent,info_dc=dc(x,function,df,res)
        computes the descent direction with parameters 
           -x: the point x
           -df : the gradient of function at point x
           -function : the function
        The function dc returns
            -descent : the direction of descent
            -info_dc : information about the behavior of the function dc   
    ls : callable
        x2,f2,df2,step2,info_ls=ls(x, function, step, descent,f,df)
        performs a line search, the parameters are
           -x : initial point
           -step : initial step
           -function : the function to be minimized
           -f,df : the values of function(x) and the gradient of the function of x
           -descent : the descent direction
        the function returns
            -x2 : the new point x+step2*descent
            -f2 : the value of function at point x2
            -df2 : the value of the gradient of the function at point x2
            -step2 : the step given by the linesearch
            -info_ls : some information about the behavior of the function ls
    itermax : int
        maximum number of iterations
    tol : float
       stopping criterion
    verbose : Boolean
        Printing option of the algorithm
    
    Returns
    --------
    The function returns a single dictionnary res, the entries are
    res['list_x'] : list of 1d vectors of size n which are the iterates points of the algorithm
    res['list_steps'] : list of positive floats which are the different steps
    res['list_grads'] : list of positive floats which are the value of the euclidean norm of the gradients of the function
    res['final_x'] : 1d vector, final value of x
    res['dc'] : list of the different infos given by the functions dc
    res['ls'] : list of the different infos given by the functions ls        
    """
    x = xini
    res={'list_x':[],'list_steps':[],'list_costs':[],'list_grads':[],'final_x':[],'dc':[],'ls':[]}
    nbiter = 0
    f=function.value(x)
    df= function.grad(x)
    err=np.linalg.norm(df)
    if verbose :  print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
    res['list_x'].append(x.copy())
    res['list_costs'].append(f)
    res['list_grads'].append(err)
    while (err > tol) and (nbiter < itermax):
        descent,info_dc = dc(x, function,df)
        x,f,df,step,info_ls = ls(x, function, step, descent,f,df)
        err = np.linalg.norm(df)
        res['list_x'].append(x.copy())
        res['list_costs'].append(f)
        res['list_grads'].append(err)
        res['list_steps'].append(step)
        res['dc'].append(info_dc)
        res['ls'].append(info_ls)
        nbiter+=1
        if verbose : print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
        if (err <= tol):
            res['final_x']=np.copy(x)
            if verbose : print("Success !!! Algorithm converged !!!")
            return res
    if verbose : print("FAILED to converge")
    return res        

def dc_gradient(x,function,df) :
    """Choice of direction of descent : GRADIENT METHOD
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        df : The actual value of the gradient
          
    returns
    -------
       descent : 1d vector of size n
           direction of descent 
       ls_info : 
           Information about the behavior of the function
    """
    descent=-df
    ls_info=None
    return descent,ls_info

#recherche linéaire pas constant
def ls_constant(x, function, step, descent,f,df) :
    """Line search : FIXED STEP
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        step : float
            The starting guess of the step
        descent : 1d vector of size n
            The descent direction
        f : float
            the value of the function at point x
        df : 1d vector of size n
            the gradient of the function at point x
          
    returns
    -------
        x2 : 1d vector of size n
            x2=x+step2*descent
        f2 : float
            the value of the function at point x2
        df2 : 1d vector of size n
            the gradient of the function at point x2
        step2 : float
            The step chosen by the method
        info : Information about the behavior of the method 
       
    """
    step2=step
    x2=x+step2*descent
    f2=function.value(x)
    df2=function.grad(x)
    info=None
    return x2,f2,df2,step2,info

def ls_backtracking(x,function,step,descent,f,df):
    step2=step
    while function.value(x+step2*descent)>= f:
        step2=step2/2
    x2=x+step2*descent
    f2=function.value(x2)
    df2=function.grad(x2)
    info=None
    return x2,f2,df2,step2,info

def ls_partial_linesearch (x,function,step,descent,f,df):
    pas=[0.1*step,0.5*step,step,2*step,10*step]
    min_val=function.value(x+pas[0]*descent)
    for i in range(1,len(pas)):
        if min_val > function.value(x+pas[i]*descent):
            min_val=function.value(x+pas[i]*descent)
    for i in range(len(pas)):
        if min_val==function.value(x+pas[i]*descent):
            step2=pas[i]
    x2=x+step2*descent
    f2=function.value(x2)
    df2=function.grad(x2)
    info=None
    return x2,f2,df2,step2,info



def ls_wolfe(x,function,step,descent,f,df,e1=1e-4,e2=0.9):
    k=0
    step2=step
    sm =0
    sp = +np.inf
    while (function.value(x+step2*descent) > function.value(x)+e1*step2*np.dot(function.grad(x),descent)) or (np.dot(function.grad(x+step2*descent),descent) < e2*np.dot(function.grad(x),descent)):
        if function.value(x+step2*descent) > function.value(x)+e1*step2*np.dot(function.grad(x),descent):
            sp=step2
            step2=(sp+sm)/2
        elif np.dot(function.grad(x+step2*descent),descent) < e2*np.dot(function.grad(x),descent):
            sm=step2
            if sp < +np.inf:
                step2=(sm+sp)/2
            else:
                step2=2*step2
        k=k+1
    x2=x+step2*descent
    f2=function.value(x2)
    df2=function.grad(x2)
    info=k
    return x2,f2,df2,step2,info
    

def dc_newton (x,function,df):
    info=None
    Hinv=np.linalg.inv(function.Hess(x))
    exp=-Hinv@df
    if (np.dot(exp,-df)/(np.linalg.norm(exp)*np.linalg.norm(-df)))>0.1:
        return exp,info
    else:
        return -df,info


def ls_wolfe_step_is_one (x,function,step,descent,f,df):
    return ls_wolfe(x,function,1,descent,f,df)


def cubic_interpolation(phi_0, phi_0_prime, phi_s, phi_s_prime, s):
    
   
    d1 = phi_0_prime + phi_s_prime - 3 * (phi_0 - phi_s) / s
    d2 = np.sign(s) * np.sqrt(d1**2 - phi_0_prime * phi_s_prime)
    c = (-phi_s_prime + d1 + d2) / (3 * s**2)
    d = (phi_s_prime - phi_0_prime - 2*d1 - 2*d2) / (3 * s**3)
    b = phi_0_prime
    a = phi_0

    return a, b, c, d

def find_minimum_of_cubic(a, b, c, d):
    
    
    coefficients = [3*d, 2*c, b]
    roots = np.roots(coefficients)
    real_roots = roots[np.isreal(roots)].real

    # min
    x_s = real_roots[np.argmin(np.polyval([d, c, b, a], real_roots))]
    return x_s
def ls_wolfe_cubique(x,function,s,descent,f,df,e1=1e-4,e2=0.9):
    s = 1


    while (function.value(x+s*descent) > function.value(x)+e1*s*np.dot(function.grad(x),descent)) or (np.dot(function.grad(x+s*descent),descent) < e2*np.dot(function.grad(x),descent)):
       
        phi_0 = function.value(x)
        phi_0_prime = np.dot(function.grad(x), descent)
        phi_s = function.value(x + s * descent)
        phi_s_prime = np.dot(function.grad(x + s * descent), descent)

       
        a, b, c, d = cubic_interpolation(phi_0, phi_0_prime, phi_s, phi_s_prime, s)
        x_s = find_minimum_of_cubic(a, b, c, d)

       
        if 0 <= x_s <= s:
            s = x_s
        else:
            s = s / 2
    x2=x+s*descent
    f2=function.value(x2)
    df2=function.grad(x2)
    info=None   
    return x2,f2,df2,s,info

