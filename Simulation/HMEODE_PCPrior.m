% Simulation 1
% Random effects were generated from the Student's t distribution;
% Measurement errors are generated from Student's t-distribution.
% Assign penalised complexity (PC) priors on kappa and nu.

% Add path
%addpath('C:\Users\baisenl\Downloads\Code_ODE_HeavyTailed\fdaM')

odeoptions = odeset('RelTol',1e-7,'AbsTol',1e-7);
Lsqnonlinoptions = optimset('Display', 'iter','Tolfun',1e-7,'Largescale','off');

%% Generate simulated data.

p  = 3;

ncohort = 50;
tobs = linspace(0,1,21)';
nobs = length(tobs);

tmin = min(tobs); tmax = max(tobs);
tEnd = tmax;

tfine    = linspace(tmin,tmax,101);
CSoluFine  = zeros(length(tfine), ncohort);

%% Generating the simulation data.

Nreps = 100;

sigma_epsilon = 0.4;
Sigma = eye(2);
Sigma(1,1) = 0.25;
Sigma(2,2) = 1;
Sigma(1,2) = 0.5*1*0.5;
Sigma(2,1) = Sigma(1,2);
[U,S,V]=svd(Sigma);
Amat = V'*sqrt(S)*U; 

truea =  3;
trueb =  10;
truepar   = [truea; trueb];
thetaImat = zeros(2,ncohort);

YobsmatArray = cell(Nreps,1);
TrueParaArray = cell(Nreps,1);
muSoluFArray = cell(Nreps,1);
mumatArray = cell(Nreps,1);

Yobsmat = zeros(nobs,ncohort);
mumat = zeros(nobs,ncohort);
muSoluF = zeros(length(tfine),ncohort);

for tt=1:Nreps
    display(num2str(tt));
        
    thetaImat=truepar*ones(1,ncohort)+Amat*sqrt((4-2)/4)*trnd(4,2,ncohort);
    
    X0 = 1+0.2/sqrt(2)*trnd(4,ncohort,1);

    for i=1:ncohort
        mytheta = thetaImat(:,i);
        mystruct1.a = mytheta(1);
        mystruct1.b = mytheta(2);

        [tobs,myfit] = ode45(@Odefun, tobs, X0(i),odeoptions,mystruct1);   
        mumat(:,i)=myfit;

        [tfine,Cfine] = ode45(@Odefun, tfine, X0(i),odeoptions,mystruct1);  
         muSoluF(:,i)=Cfine;
    end
    
    true_par = zeros(3,ncohort);
    true_par(1,:)=X0';
    true_par(2:3,:)=thetaImat;
    
    TrueParaArray{tt} = true_par;
    
    muSoluFArray{tt}=muSoluF;
    mumatArray{tt}=mumat;

end

for tt=1:Nreps
    
mumat = mumatArray{tt};
Yobsmat = mumat+sqrt((4-2)/4)*trnd(4,nobs,ncohort);
YobsmatArray{tt} = Yobsmat;

end

%% Preparing for MCMC algorithm.

knots   = linspace(0,tEnd,nobs)';
norder  = 4;
nbasis  = length(knots) + norder - 2;
basis   = create_bspline_basis([0,tEnd],nbasis,norder,knots);
J       = nbasis; 

% basis functions evaluated at tobs
Phimat   = eval_basis(tobs, basis);

% first derivative of basis functions evaluated at tobs
D1Phimat = eval_basis(tobs, basis, 1);
% second derivative of basis functions evaluated at tobs
D2Phimat = eval_basis(tobs, basis, 2);

% basis functions evaluated at tobs
Phimat_fine   = eval_basis(tfine, basis);

% first derivative of basis functions evaluated at tobs
D1Phimat_fine = eval_basis(tfine, basis, 1);
% second derivative of basis functions evaluated at tobs
D2Phimat_fine = eval_basis(tfine, basis, 2);


%% Calculate penalty term using Simpson's rule
h = 0.01;
quadpts = (0:h:tEnd)';
nquad = length(quadpts);
quadwts = ones(nquad,1);
quadwts(2:2:(nquad-1)) = 4;
quadwts(3:2:(nquad-2)) = 2;
quadwts = quadwts.*(h/3);

D0Qbasismat = eval_basis(quadpts, basis);
D1Qbasismat = eval_basis(quadpts, basis, 1);
D2Qbasismat = eval_basis(quadpts, basis, 2);
R1mat = D1Qbasismat'*(D1Qbasismat.*(quadwts*ones(1,nbasis)));
R2mat = D2Qbasismat'*(D2Qbasismat.*(quadwts*ones(1,nbasis)));

%% Set optimal criteria

options1 = optimset( ...
    'Algorithm', 'levenberg-marquardt', ...
    'LargeScale', 'off', ...
    'Display','off', ...
    'DerivativeCheck','off', ...
    'MaxIter', 100, ...
    'Jacobian','off',...
    'MaxFunEvals', 100000,...
    'TolFun', 1e-10, ...
    'TolX',   1e-10);

%% Start MCMC algorithm.

m_N = zeros(2,Nreps);
m_ST = zeros(2,Nreps);
thetaEst_NN=zeros(2,Nreps);
thetaEst_TT=zeros(2,Nreps);

Sim1_thetaSam_TTArray = cell(Nreps,1);
Sim1_thetaSam_NNArray = cell(Nreps,1);
Sim1_thetaISam_TTArray = cell(Nreps,1);
Sim1_thetaISam_NNArray = cell(Nreps,1);
Sim1_sigmaESam_TTArray = cell(Nreps,1);
Sim1_sigmaESam_NNArray = cell(Nreps,1);
Sim1_InvSigmaSam_TTArray = cell(Nreps,1);
Sim1_InvSigmaSam_NNArray = cell(Nreps,1);
Sim1_USam_TTArray = cell(Nreps,1);
Sim1_WSam_TTArray = cell(Nreps,1);
Sim1_kappaSam_TTArray = cell(Nreps,1);
Sim1_nuSam_TTArray = cell(Nreps,1);
Sim1_muSam_TTArray = cell(Nreps,1);
Sim1_muSam_NNArray = cell(Nreps,1);
Sim1_cparmatSam_TTArray = cell(Nreps,1);
Sim1_cparmatSam_NNArray = cell(Nreps,1);


%%

ITER = 5e3;
Final_size = 500;

sigmaESam_TT  = zeros(Final_size,1); 
thetaSam_TT      = zeros(p,Final_size);
thetaISam_TT     = zeros(p,ncohort,Final_size);
InvSigmaSam_TT  = zeros(p,p,Final_size);
USam_TT = zeros(ncohort,Final_size);
WSam_TT = zeros(ncohort,Final_size);
kappaSam_TT = zeros(Final_size,1);
nuSam_TT = zeros(Final_size,1);
muSam_TT = zeros(nobs,ncohort,Final_size);
cparmatSam_TT = zeros(J,ncohort,Final_size);

sigmaESam_NN  = zeros(Final_size,1); 
thetaSam_NN      = zeros(p,Final_size);
thetaISam_NN     = zeros(p,ncohort,Final_size);
InvSigmaSam_NN  = zeros(p,p,Final_size);
muSam_NN = zeros(nobs,ncohort,Final_size);
cparmatSam_NN = zeros(J,ncohort,Final_size);

for tt=1:Nreps
  
    display(num2str(tt));
    
true_par = TrueParaArray{tt};
muSoluF = muSoluFArray{tt};
mumat = mumatArray{tt};
Yobsmat = YobsmatArray{tt};

thetaIInt=true_par;

%% MCMC algorithm for Student-t/Student-t model.

df  = p+1;
a   = 1; % shape parameter of gamma distribution.
b   = 10; % rate parameter of gamma distribution.
c   = 0.02;
d   = 10;

S01   = 0.01*eye(p); 
eta0     = [1; 3; 10]; 
InvOmega = 0.01*eye(p);
thetaInt = eta0+0.0000*normrnd(0, abs(eta0),p,1); 
truethetaI = thetaIInt;
InvSigma1Int = diag(exp(normrnd(-1,1,p,1)));


% Initial values;

oldcparmat = (Phimat'*Phimat+0.0001*eye(J))\(Phimat'*mumat);
oldmumat  = mumat;
oldsigmaE   = 1;
oldkappa = 3;
oldnu = 3;

oldthetaImat  = truethetaI;
oldtheta      = thetaInt;
oldInvSigma   = InvSigma1Int;
oldUmat       = 3*ones(ncohort,1);
oldWmat       = 3*ones(ncohort,1);

%  proposal;

sig1 = 0.3*[0.2;0.25; 0.25]; % For theta_i

sig2 = 0.5; % For kappa
sig3 = 0.5; % For nu

%%%%%%
% Gibbs sampler.

countthetaI = zeros(ncohort,1);
countkappa  = 0;
countnu     = 0;

%%%%%%%%%%%
%% Start MCMC algorithm.


input1.oldcparmat   = oldcparmat;
input1.oldthetaImat = oldthetaImat;
input1.oldtheta     = oldtheta;
input1.oldmumat     = oldmumat;
input1.oldsigmaE   = oldsigmaE;
input1.oldkappa = oldkappa;
input1.oldnu  = oldnu;

input1.oldUmat   = oldUmat;
input1.oldWmat   = oldWmat;

input1.sig1 = sig1;
input1.sig2 = sig2;
input1.sig3 = sig3;

input1.p    = p;

input1.ncohort = ncohort;
input1.Yobsmat = Yobsmat;
input1.tobs = tobs;

input1.Phimat = Phimat;
input1.D0Qbasismat = D0Qbasismat;
input1.D1Qbasismat = D1Qbasismat;
input1.quadwts = quadwts;

input1.eta0    = eta0;
input1.S01     = S01;
input1.df      = df;
input1.a       = a;
input1.b       = b;
input1.c       = c;
input1.d       = d;
input1.lambda_kappa = 5.1;
input1.lambda_nu    = 5.1;

input1.oldInvSigma = oldInvSigma;
input1.InvOmega    = InvOmega;


input1.countthetaI = countthetaI;
input1.countkappa  = countkappa;
input1.countnu     = countnu;

input1.options1    = options1;


out1 = PCPrior_MCMC_funTT(input1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% burn-in samples.


for iter=1:ITER
    %display(num2str(iter))

    input1 = out1;
    
    out1 = PCPrior_MCMC_funTT(input1);
    
    oldthetaImat = out1.oldthetaImat;
    oldtheta     = out1.oldtheta;
    oldInvSigma  = out1.oldInvSigma;
    oldsigmaE    = out1.oldsigmaE;
    oldmumat     = out1.oldmumat;
    oldcparmat   = out1.oldcparmat;
    
    oldUmat      = out1.oldUmat;
    oldWmat      = out1.oldWmat;
    oldkappa     = out1.oldkappa;
    oldnu        = out1.oldnu;
    
    countthetaI  = out1.countthetaI;
    countkappa   = out1.countkappa;
    countnu      = out1.countnu;
  
end


%% Final samples

ctr1 = 1;

for iter=1:ITER
    %display(num2str(iter))

    input1 = out1;
    
    out1 = PCPrior_MCMC_funTT(input1);
    
    oldthetaImat = out1.oldthetaImat;
    oldtheta     = out1.oldtheta;
    oldInvSigma  = out1.oldInvSigma;
    oldsigmaE    = out1.oldsigmaE;
    oldmumat     = out1.oldmumat;
    oldcparmat   = out1.oldcparmat;
    
    oldUmat      = out1.oldUmat;
    oldWmat      = out1.oldWmat;
    oldkappa     = out1.oldkappa;
    oldnu        = out1.oldnu;   
    
    countthetaI  = out1.countthetaI;
    countkappa   = out1.countkappa;
    countnu      = out1.countnu;
    
  if mod(iter,10)==0   
    
    thetaISam_TT(:,:,ctr1) = oldthetaImat;
    thetaSam_TT(:,ctr1)    = oldtheta;
    InvSigmaSam_TT(:,:,ctr1) = oldInvSigma;
    
    sigmaESam_TT(ctr1)       = oldsigmaE;
    USam_TT(:,ctr1)        = oldUmat;
    WSam_TT(:,ctr1)        = oldWmat;
    kappaSam_TT(ctr1)      = oldkappa;
    nuSam_TT(ctr1)         = oldnu;

    muSam_TT(:,:,ctr1)       = oldmumat;
    cparmatSam_TT(:,:,ctr1)  = oldcparmat;    
    
   ctr1 = ctr1+1;
    
  end

 if mod(iter,20)==20 
        figure(13)

        for ii=1:25
            subplot(5,5,ii);
            plot(tobs,Yobsmat(:,ii), 'ko', tobs, oldmumat(:,ii),'r-');

            xlabel('Time');
            ylabel('C(t)');
            title(['Patient',num2str(ii)])
            xlim([0 tEnd]);
        end
 end

end


Sim1_thetaSam_TTArray{tt}  = thetaSam_TT;
Sim1_thetaISam_TTArray{tt} = thetaISam_TT;
Sim1_InvSigmaSam_TTArray{tt} = InvSigmaSam_TT;    
Sim1_sigmaESam_TTArray{tt} =    sigmaESam_TT;
Sim1_USam_TTArray{tt} =    USam_TT;
Sim1_WSam_TTArray{tt} =    WSam_TT;
Sim1_kappaSam_TTArray{tt} = kappaSam_TT;
Sim1_nuSam_TTArray{tt}    = nuSam_TT;
Sim1_muSam_TTArray{tt}    = muSam_TT;
Sim1_cparmatSam_TTArray{tt} =  cparmatSam_TT;

countthetaI/ITER/2;
countkappa/ITER/2;
countnu/ITER/2;


display(num2str('End of SMN model'));

%% MCMC algorithm for Normal/Normal model.

countthetaI = zeros(ncohort,1);

input1.oldcparmat   = oldcparmat;
input1.oldthetaImat = oldthetaImat;
input1.oldtheta     = oldtheta;
input1.oldmumat     = oldmumat;
input1.oldsigmaE   = oldsigmaE;

input1.sig1 = sig1;

input1.p    = p;

input1.ncohort = ncohort;
input1.Yobsmat = Yobsmat;
input1.tobs = tobs;

input1.Phimat = Phimat;
input1.D0Qbasismat = D0Qbasismat;
input1.D1Qbasismat = D1Qbasismat;
input1.quadwts = quadwts;

input1.eta0    = eta0;
input1.S01     = S01;
input1.df      = df;
input1.a       = a;
input1.b       = b;
input1.c       = c;
input1.d       = d;

input1.oldInvSigma = oldInvSigma;
input1.InvOmega    = InvOmega;

input1.countthetaI = countthetaI;

input1.options1    = options1;


out1 = MCMC_funNN(input1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% burn-in samples.

for iter=1:ITER
   % display(num2str(iter))

   
    input1 = out1;
    
    out1 = MCMC_funNN(input1);
    
    oldthetaImat = out1.oldthetaImat;
    oldtheta     = out1.oldtheta;
    oldInvSigma  = out1.oldInvSigma;
    oldsigmaE    = out1.oldsigmaE;
    oldmumat     = out1.oldmumat;
    oldcparmat   = out1.oldcparmat;
       
    countthetaI  = out1.countthetaI;
      
end

%% Final samples.


ctr2 = 1;

for iter=1:ITER
   % display(num2str(iter))

   
    input1 = out1;
    
    out1 = MCMC_funNN(input1);
    
    oldthetaImat = out1.oldthetaImat;
    oldtheta     = out1.oldtheta;
    oldInvSigma  = out1.oldInvSigma;
    oldsigmaE    = out1.oldsigmaE;
    oldmumat     = out1.oldmumat;
    oldcparmat   = out1.oldcparmat;
       
    countthetaI  = out1.countthetaI;
  
    
    if mod(iter,10)==0
    
        thetaISam_NN(:,:,ctr2) = oldthetaImat;
        thetaSam_NN(:,ctr2)    = oldtheta;
        InvSigmaSam_NN(:,:,ctr2) = oldInvSigma;
        
        sigmaESam_NN(ctr2)       = oldsigmaE;

        muSam_NN(:,:,ctr2)       = oldmumat;
        cparmatSam_NN(:,:,ctr2)  = oldcparmat;
        
        ctr2 = ctr2+1;
    
     end

 if mod(iter,20)==20 
        figure(13)

        for ii=1:25
            subplot(5,5,ii);
            plot(tobs,Yobsmat(:,ii), 'ko', tobs, oldmumat(:,ii),'r-');

            xlabel('Time');
            ylabel('C(t)');
            title(['Patient',num2str(ii)])
            xlim([0 tEnd]);
        end
 end

end


Sim1_thetaSam_NNArray{tt}  = thetaSam_NN;
Sim1_thetaISam_NNArray{tt} = thetaISam_NN;
Sim1_InvSigmaSam_NNArray{tt} = InvSigmaSam_NN;    
Sim1_sigmaESam_NNArray{tt}  = sigmaESam_NN;
Sim1_muSam_NNArray{tt}      = muSam_NN;
Sim1_cparmatSam_NNArray{tt} = cparmatSam_NN;

display(num2str('End of normal model'));

%%
truetheta = [3;10];

temp1=Sim1_thetaSam_NNArray{tt};
temp2=Sim1_thetaSam_TTArray{tt};
thetaEst_NN(:,tt) = mean(temp1(2:3,:),2);
thetaEst_TT(:,tt) = mean(temp2(2:3,:),2);

if(tt>3)
    mean_thetaEst_NN=mean(thetaEst_NN(:,1:tt),2);
    mean_thetaEst_TT=mean(thetaEst_TT(:,1:tt),2);
    SD_thetaEst_NN = sqrt(diag(cov(thetaEst_NN(:,1:tt)')));
    SD_thetaEst_TT = sqrt(diag(cov(thetaEst_TT(:,1:tt)')));
    MADE_thetaEst_NN = mean(abs(thetaEst_NN(:,1:tt)-truetheta*ones(1,tt)),2);
    MADE_thetaEst_TT = mean(abs(thetaEst_TT(:,1:tt)-truetheta*ones(1,tt)),2);
    result_Sim1 = [mean_thetaEst_NN mean_thetaEst_TT SD_thetaEst_NN SD_thetaEst_TT MADE_thetaEst_NN MADE_thetaEst_TT]

end


end

%% Prediction accurarcy.

MAPE_TT1 = zeros(Nreps,1);
MAPE_TT2 = zeros(Nreps,1);
MAPE_NN1 = zeros(Nreps,1);
MAPE_NN2 = zeros(Nreps,1);
MSPE_TT1 = zeros(Nreps,1);
MSPE_TT2 = zeros(Nreps,1);
MSPE_NN1 = zeros(Nreps,1);
MSPE_NN2 = zeros(Nreps,1);


for tt=1:Nreps
newtt = 0:0.01:3;
newmumat = zeros(length(newtt),ncohort);
newmumat_TT = zeros(length(newtt),ncohort);
newmumat_NN = zeros(length(newtt),ncohort);


mean_thetaImat_TT = zeros(3,ncohort);
mean_thetaImat_NN = zeros(3,ncohort);

thetaISam_TT = Sim1_thetaISam_TTArray{tt};
thetaISam_NN = Sim1_thetaISam_NNArray{tt};



for ctr1=1:Final_size
    mean_thetaImat_TT=mean_thetaImat_TT+thetaISam_TT(:,:,ctr1);
    mean_thetaImat_NN=mean_thetaImat_NN+thetaISam_NN(:,:,ctr1);
end
mean_thetaImat_TT = 1/Final_size*mean_thetaImat_TT;
mean_thetaImat_NN = 1/Final_size*mean_thetaImat_NN;

true_par = TrueParaArray{tt};


for i=1:ncohort
        mytheta = true_par(2:3,i);
        mystruct1.a = mytheta(1);
        mystruct1.b = mytheta(2);

        [newtt,myfit] = ode45(@Odefun, newtt, true_par(1,i),odeoptions,mystruct1);   
        newmumat(:,i)=myfit;

        mytheta = mean_thetaImat_TT(2:3,i);
        mystruct1.a = mytheta(1);
        mystruct1.b = mytheta(2);

        [newtt,myfit] = ode45(@Odefun, newtt, mean_thetaImat_TT(1,i),odeoptions,mystruct1);   
        newmumat_TT(:,i)=myfit;

        mytheta = mean_thetaImat_NN(2:3,i);
        mystruct1.a = mytheta(1);
        mystruct1.b = mytheta(2);

        [newtt,myfit] = ode45(@Odefun, newtt, mean_thetaImat_NN(1,i),odeoptions,mystruct1);   
        newmumat_NN(:,i)=myfit;

end

MAPE_TT1(tt) = 1/(ncohort*length(newtt))*sum(sum(abs(newmumat-newmumat_TT)));
MAPE_NN1(tt) = 1/(ncohort*length(newtt))*sum(sum(abs(newmumat-newmumat_NN)));

MAPE_TT2(tt) = 1/(ncohort*length(newtt))*sum(sum(abs(newmumat(101:301,:)-newmumat_TT(101:301,:))));
MAPE_NN2(tt) = 1/(ncohort*length(newtt))*sum(sum(abs(newmumat(101:301,:)-newmumat_NN(101:301,:))));

MSPE_TT1(tt) = 1/(ncohort*length(newtt))*sum(sum((newmumat-newmumat_TT).^2));
MSPE_NN1(tt) = 1/(ncohort*length(newtt))*sum(sum((newmumat-newmumat_NN).^2));

MSPE_TT2(tt) = 1/(ncohort*length(newtt))*sum(sum((newmumat(101:301,:)-newmumat_TT(101:301,:)).^2));
MSPE_NN2(tt) = 1/(ncohort*length(newtt))*sum(sum((newmumat(101:301,:)-newmumat_NN(101:301,:)).^2));


end


[mean(MAPE_TT1) mean(MAPE_NN1) mean(MAPE_TT2) mean(MAPE_NN2)  ]
[std(MAPE_TT1) std(MAPE_NN1) std(MAPE_TT2) std(MAPE_NN2)  ]

[mean(MSPE_TT1) mean(MSPE_NN1) mean(MSPE_TT2) mean(MSPE_NN2)  ]
[std(MSPE_TT1) std(MSPE_NN1) std(MSPE_TT2) std(MSPE_NN2)  ]

%% End.



        