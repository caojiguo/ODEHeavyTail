# Matlab codes for Bayesian Inference of Mixed-effects Ordinary Differential Equations Models Using Heavy-tailed Distributions

This repository contains the Matlab codes for the simulation studies in Section 5 of the article "Bayesian Inference of Mixed-effects Ordinary Differential Equations Models Using Heavy-tailed Distributions" by Liu, Wang, Nie and Cao (2018)

There are totally two folders: "Functions" and "Simulations".

The folder "Functions" includes the Matlab functions of MCMC algorithms for SMN model and normal model.

The folder "Simulations" includes the main Matlab codes to simulation.

==================== Simulations ==============================================


HMEODE_T.m: the main Matlab code to simulate the hierarchical mixed-effects ODE model, where the random-effects are generated from the Student's t-distribution.

HMEODE_GeneralizedHyper.m: the main Matlab code to simulate the hierarchical mixed-effects ODE model, where the random-effects are generated from the generalized hyperbolic distribution (GH).

HMEODE_MixtureT.m: the main Matlab code to simulate the hierarchical mixed-effects ODE model, where the random-effects are generated from the mixture of Student's t-distributions (MixT).

HMEODE_InverseGaussian.m: the main Matlab code to simulate the hierarchical mixed-effects ODE model, where the random-effects are generated from the inverse Gaussian distribution (IG).

HMEODE_BSSN.m: the main Matlab code to simulate the hierarchical mixed-effects ODE model, where the random-effects are generated from the Birnbaum-Saunders distribution (BSSN).

HMEODE_HalfNormal.m: the main Matlab code to simulate the hierarchical mixed-effects ODE model, where a half-normal prior is assigned on the standard deviation parameter, sigma_epsilon.

HMEODE_PCPrior.m: the main Matlab code to simulate the hierarchical mixed-effects ODE model, where the penalised complexity (PC) priors are assigned on kappa and nu in SMN model.

GHyperbolic.R: the R code to generate the random numbers of generalized hyperbolic distributions  by using the R package "GeneralizedHyperbolic".

GH_theta1Sam.mat: the simulated values of individual parameter, theta_{i1},  using R package "GeneralizedHyperbolic".
GH_theta2Sam.mat: the simulated values of individual parameter, theta_{i2}, using R package "GeneralizedHyperbolic".
GH_theta1Sam.mat: the simulated values of initial conditions, X_{i}(0), using R package "GeneralizedHyperbolic".
 

In the above simulations, the measurement errors are all generated from the Student's t-distribution.

====================Functions==============================================


MCMC_funNN.m: the MCMC algorithm assuming that the random-effects and errors follow Gaussian distributions, and assigning a inverse gamma prior on the variance parameter (sigma_epsilon*sigma_epsilon).

HalfNormal_MCMC_funNN.m: the MCMC algorithm assuming that the random-effects and errors follow Gaussian distributions, and assigning a half-normal prior on the standard deviation parameter sigma_epsilon.

MCMC_funTT.m: the MCMC algorithm assuming that the random-effects and errors follow heavy-tailed distributions, and assigning a inverse gamma prior on the variance parameter (sigma_epsilon*sigma_epsilon).

HalfNormal_MCMC_funTT.m: the MCMC algorithm assuming that the random-effects and errors follow heavy-tailed distributions, and assigning a inverse gamma prior on the standard deviation parameter sigma_epsilon.

PCPrior_MCMC_funTT.m: the MCMC algorithm assuming that the random-effects and errors follow heavy-tailed distributions, and assigning  penalised complexity (PC) priors on the kappa and nu.

PCprior.m: to calculate the density of PC priors.

Odefun.m: the ordinary differential equations used in Simulation 1.

LSQNONLIN_fun.m: to solve the ode numerically.


==============================================================

Note:  (1) Before performing this simulation,  you need to download the FDA package to use it.
       (2) Contact information: jiguo_cao@sfu.ca. or bsliu2014@dufe.edu.cn.

 
