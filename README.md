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

      - **Expected improvement (EI)** (also MEI?) - should confirm
        - Looks where a function **may most improve**, aka *maximal expected utility*
        - EI(x)=𝔼[max{0,f(x)−f(x̂ )}]
        - Crowd favorite
      
      - **Entropy search**
        - Improves function by **minimizing the uncertainty** of any predicted optimium.
    
      - **Upper Confidence Bound** (UCB)
        - Looks where a function's **improvement is most likely**
        - Looks to exploits possibly uncertainty by finding where the upperbound may be undetermined.
        
      - **Maximum probability of improvement** (MPI)
      
      - **PMAX**
      - **IEMAX**
      - **GP-Hedge**

   As to follow the crowd this notebook will use the Expected Improvement function, for reasons I may revisit this notebook to explain.
    
  
   With these two parts our program should:
   ### Pseudo-code:
   - Given observed values of target function f(x), update the posterior expectation of f using the Gaussian Process.
   - Find new_x that maximises the ```EI: new_x = argmax EI(x)```.
   - Compute the value of f(new_x).
   - Update posterior expectation of f
        - Repeat this process for n_iterations
  
  
## Example Function
As we will ultimately be looking at hyperparameters, by treating the score or error as a function of the parameters.
In this case we treat it as an optimization problem for an example function, finding the global *minimum* (for error).

![example eq_1](https://user-images.githubusercontent.com/36013672/41205184-88fddb80-6cbc-11e8-8cbe-c704ecb50a68.png)

![true](https://user-images.githubusercontent.com/36013672/41138347-b7a024ee-6aae-11e8-8b7e-b45bd660d4de.png)


### GridSearch & RandomSearch 
Using 15 points, finding minimums.

![grid](https://user-images.githubusercontent.com/36013672/41138351-bb5d08b8-6aae-11e8-8fd1-ecbcfe5bdd3c.png)
![random](https://user-images.githubusercontent.com/36013672/41138354-bcf1d276-6aae-11e8-8aa0-f4965a37134a.png)

GridSearching misses the true global min, as true existed outside of discrete boundaries
RandomSearch also misses true global min, but could have been better than Grid

### Bayesian Opt
Generating priors(inital 5)

![bay_5](https://user-images.githubusercontent.com/36013672/41138356-c0ba6102-6aae-11e8-8dd6-f3e1db8ce08c.png)

Updating process with each aquisition.
*Note the red line maps to the aquisition function on the bottom and refer to the where to look next*

![bay_6](https://user-images.githubusercontent.com/36013672/41138357-c0c9a9dc-6aae-11e8-97ed-99d02f08f03e.png)
![bay_7](https://user-images.githubusercontent.com/36013672/41138358-c0d58cfc-6aae-11e8-9f76-5d51071d4819.png)
![bay_8](https://user-images.githubusercontent.com/36013672/41138359-c0e71440-6aae-11e8-888a-d40aa36069cb.png)
![bay_9](https://user-images.githubusercontent.com/36013672/41138360-c0f5dd18-6aae-11e8-9b23-5a509f72b573.png)
![bay_10](https://user-images.githubusercontent.com/36013672/41138361-c104613a-6aae-11e8-8781-14f15414d00e.png)
![bay_11](https://user-images.githubusercontent.com/36013672/41138362-c110e7fc-6aae-11e8-9d5f-e44ffdd1a6af.png)
![bay_12](https://user-images.githubusercontent.com/36013672/41138363-c11d251c-6aae-11e8-98f9-7f4b93c9c9b4.png)
![bay_13](https://user-images.githubusercontent.com/36013672/41138364-c12c62d4-6aae-11e8-963e-1047b09d1305.png)
![bay_14](https://user-images.githubusercontent.com/36013672/41138365-c13b136a-6aae-11e8-85b9-a7d670218d3e.png)
![bay_15](https://user-images.githubusercontent.com/36013672/41138366-c14907c2-6aae-11e8-9bc2-754d28058157.png)

Bay finds global optimum, confirms it is global, and searchs for the true.

### Comparing methods from 15 Points

![grid](https://user-images.githubusercontent.com/36013672/41138351-bb5d08b8-6aae-11e8-8fd1-ecbcfe5bdd3c.png)
![random](https://user-images.githubusercontent.com/36013672/41138354-bcf1d276-6aae-11e8-8aa0-f4965a37134a.png)
![bay_15](https://user-images.githubusercontent.com/36013672/41138366-c14907c2-6aae-11e8-9bc2-754d28058157.png)
![true](https://user-images.githubusercontent.com/36013672/41138347-b7a024ee-6aae-11e8-8b7e-b45bd660d4de.png)

- Gridsearching would search the param space symetrically and systematically.
    - thorough, inefficient, uniformity between samples may miss details.
        - weak in higher dimensional space
- Randomsearching would search the param space randomly.
    - efficient, less thorough, reliant on sufficent iterations
        - stronger in higher dimensional space 

*neither learn from previously selected elements in the parameter space.*
- Bayesian however, does learn from previous elements, and works effectively with increased dimensional space.



# 2 Parameters
## Example Function
Again, by treating the score or error as a function of the parameters.
In this case we treat it as an optimization problem for an example function, finding the global *maximum* (for score).
In this particular example, there are 2 maximums, which otherwise with grid and random searched would lead to confusion.

![6_hump_eq](https://user-images.githubusercontent.com/36013672/41205096-48780f50-6cbb-11e8-96de-e0046a23a7c8.png)

![camel_true](https://user-images.githubusercontent.com/36013672/41205098-4ab3428a-6cbb-11e8-8a50-31b00a05651c.png)


### GridSearch & RandomSearch 
Using 25 points, finding minimums.

![camel_grid](https://user-images.githubusercontent.com/36013672/41205110-62cd2ebc-6cbb-11e8-9ac3-084dc84088be.png)
![camel_random](https://user-images.githubusercontent.com/36013672/41205111-64d044d8-6cbb-11e8-9c6c-1541aed6079c.png)

### Bayesian Opt
Generating priors (inital 5)

![camel_5](https://user-images.githubusercontent.com/36013672/41205114-705fc22e-6cbb-11e8-9aee-74857b4c50ee.png)

Updating process with each aquisition.
*Note the point on the aquisition map and refer to the where to look next*

Bayesian (10 Points)
![camel_10](https://user-images.githubusercontent.com/36013672/41205115-70734416-6cbb-11e8-9658-8ee3288aa58b.png)

Bayesian (15 Points)
![camel_15](https://user-images.githubusercontent.com/36013672/41205116-708de474-6cbb-11e8-8fde-4faf37bb8ae6.png)

Bayesian (20 Points) -  At this point, the assumed function is fairly accurate to the true
![camel_20](https://user-images.githubusercontent.com/36013672/41205117-709dabf2-6cbb-11e8-9156-f48432e44711.png)

Bayesian (25 Points)
![camel_25](https://user-images.githubusercontent.com/36013672/41205118-70b0dfd8-6cbb-11e8-94b0-f95f30002d2d.png)

### Comparing Methods with 25 points

![camel_grid](https://user-images.githubusercontent.com/36013672/41205110-62cd2ebc-6cbb-11e8-9ac3-084dc84088be.png)
![camel_random](https://user-images.githubusercontent.com/36013672/41205111-64d044d8-6cbb-11e8-9c6c-1541aed6079c.png)
![camel_25](https://user-images.githubusercontent.com/36013672/41205118-70b0dfd8-6cbb-11e8-94b0-f95f30002d2d.png)
![camel_true](https://user-images.githubusercontent.com/36013672/41205098-4ab3428a-6cbb-11e8-8a50-31b00a05651c.png)

Again...
- Gridsearching would search the param space symetrically and systematically.
    - thorough, inefficient, uniformity between samples may miss details.
        - weak in higher dimensional space
- Randomsearching would search the param space randomly.
    - efficient, less thorough, reliant on sufficent iterations
        - stronger in higher dimensional space 

*neither learn from previously selected elements in the parameter space.*
- Bayesian however, does learn from previous elements, and works effectively with increased dimensional space.
    #### At 20 aq points with the bayesian map, the theorized model closely resembles the true function, as compared grid & random with 25 points.
    
    
### Resources:
https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083
https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
https://github.com/fmfn/BayesianOptimization
https://sheffieldml.github.io/GPyOpt/
