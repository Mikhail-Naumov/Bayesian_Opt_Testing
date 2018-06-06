# Bayesian_Opt_Testing
### Simple Sample Baye Opt
    While a number of production ready algorithms exist such as:
    - MOE,
    - Spearmint,
    - hyperopt,
    - GPyOpt,
    
### This notebook handles making a simple algorithm.
### I did not come up with this algorithm, I am simply exploring its functionality!

  Simply put, the Bayesian Optimizers requires:
  
   **A Gaussian process**- ```sklearn.gaussian_process as gp```
   - From which we will predict the posterior distribution of the target function (function of errors for models).
   - Because a Gaussian distribution is defined by its mean and variance, this model relies on the assumption that the Gaussian process is also completely defined by its mean function and variance function.
        - Until math has another revolution and we discover that we know nothing about math (which I seem to find a lot) we can assume this is a pretty safe assumption (GP ~ mean function & variance function).\
   
        ```A GP is a popular probability model, because it induces a posterior distribution over the loss function that is analytically tractable. This allows us to update our beliefs of what looks like, after we have computed the loss for a new set of hyperparameters.``` https://thuijskens.github.io/2016/12/29/bayesian-optimisation/



   **An Aquisition function**
   
    - Which decides at which point in our target function, we want to sample next.
    - A number of well documented aquistion functions exist, listed below:
   
   
   
   **Probability of Improvement**
   
    - Looks where a function's **improvement is most likely**
    - Can lead to odd behavior because it relies on the current minimum, rather than the magnatude of possiblity of improvement.

      - **Expected improvement (EI)**
        - Looks where a function **may most improve**, aka *maximal expected utility*
        - EI(x)=𝔼[max{0,f(x)−f(x̂ )}]
        - Crowd favorite
      
      - **Entropy search**
        - Improves function by **minimizing the uncertainty** of any predicted optimium.
    
      - **Upper Confidence Bound** (UCB)
        - Looks where a function's **improvement is most likely**
        - Looks to exploits possibly uncertainty by finding where the upperbound may be undetermined.

   As to follow the crowd this notebook will use the Expected Improvement function, for reasons I may revisit this notebook to explain.
    
  
   With these two parts our program should:
   ### Pseudo-code:
   - Given observed values of target function f(x), update the posterior expectation of f using the Gaussian Process.
   - Find new_x that maximises the ```EI: new_x = argmax EI(x)```.
   - Compute the value of f(new_x).
   - Update posterior expectation of f
        - Repeat this process for n_iterations
