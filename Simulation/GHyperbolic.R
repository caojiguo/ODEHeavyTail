# This R code is used to generate the ODE parameters (theta_1, theta_2) and initial condition X(0) from the generalized hyperbolic distribution


#-----------------------Generalized hyperbolic distribution----------------#
library(GeneralizedHyperbolic);

delta <-1;
alpha <-5;
    
n <- 50;
Nreps <- 100;

truepar <- c(3,10)

Sigma <-diag(2);
Sigma[1,1] <-0.5^2;
Sigma[2,2] <-1;
Sigma[1,2] <- Sigma[2,1] <- 0.5*1*0.5;

temp <- svd(Sigma);
Amat <-t(temp$v)%*%diag(sqrt(temp$d))%*%temp$u; 

theta1Sam <- matrix(0,n,Nreps); # theta_1
theta2Sam <- matrix(0,n,Nreps); # theta_2
X0Sam <- matrix(0,n,Nreps);  # X(0)

set.seed(123457)

for(i in 1:Nreps){

    gam   <-alpha;
    temp1 <-matrix(rghyp(2*n,mu=0,delta=delta,alpha=alpha,beta=0,lambda=1),2,n);
    sigma <-sqrt(delta*besselK(alpha*delta,2)/gam/besselK(alpha*delta,1));
    temp2 <-truepar+Amat%*%temp1/sigma;
    theta1Sam[,i] <- temp2[1,]
    theta2Sam[,i] <- temp2[2,]    
    
    X0Sam[,i] <- 1+0.2/sigma*rghyp(n,mu=0,delta=delta,alpha=alpha,beta=0,lambda=1)   
    
}

 
write(t(theta1Sam),file="D:/HMEODE/Simulation/GH_theta1Sam.txt",ncolumns=Nreps);
write(t(theta2Sam),file="D:/HMEODE/Simulation/GH_theta2Sam.txt",ncolumns=Nreps);
write(t(X0Sam),file="D:/HMEODE/Simulation/GH_X0Sam.txt",ncolumns=Nreps);
    