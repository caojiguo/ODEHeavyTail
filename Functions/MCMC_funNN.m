function out1 = MCMC_funNN(input1)
% Note: this program is built on Matlab (at least 2014 version) due to
% truncated distributed are used.
% Assuming normal distributions on both random-effects and observations.
% Inverse gamma prior assigned on variance parameter.
       
        ncohort      = input1.ncohort;
        oldthetaImat = input1.oldthetaImat;
        oldtheta     = input1.oldtheta;
        oldcparmat   = input1.oldcparmat;
 
        oldsigmaE    = input1.oldsigmaE;
 
        oldInvSigma = input1.oldInvSigma;

        sig1    = input1.sig1;

        p       = input1.p;
        tobs    = input1.tobs;
        nobs    = length(tobs);

        Yobsmat = input1.Yobsmat;
        Phimat  = input1.Phimat;
        quadwts = input1.quadwts;
        
        eta0    = input1.eta0;
        S01     = input1.S01;
        df      = input1.df;
        a       = input1.a;
        b       = input1.b;  
        c       = input1.c;  
        d       = input1.d;  
        
        
        D0Qbasismat = input1.D0Qbasismat;
        D1Qbasismat = input1.D1Qbasismat;
        InvOmega    = input1.InvOmega;
         
        countthetaI = input1.countthetaI;
       
        options1 = input1.options1;
 
        %% Update theta_i
        
    for ii=1:ncohort

        oldthetaI   = oldthetaImat(:,ii);   
        newthetaI   = oldthetaI+normrnd(0,sig1,p,1);

        Yobs = Yobsmat(:,ii);

        myfitstr1.D0Qbasismat   = D0Qbasismat;
        myfitstr1.D1Qbasismat   = D1Qbasismat;
        myfitstr1.quadwts       = quadwts;
        
        myfitstr1.c0            = oldthetaI(1);
        
        myfitstr1.thetaI        = oldthetaI(2:p);
        oldcpar                 = oldcparmat(:,ii);
   
        oldcpar = lsqnonlin(@LSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        oldcpar(1) = oldthetaI(1);
 
        oldcparmat(:,ii) = oldcpar;
        
        oldmu  = Phimat*oldcpar; 

        olddiff  = (Yobs-oldmu);  
        oldSSE   = -0.5/oldsigmaE*sum(olddiff.^2);
        oldprior = -0.5*(oldthetaI-oldtheta(1:p))'*oldInvSigma*(oldthetaI-oldtheta(1:p));
        den1     = oldSSE + oldprior;


        myfitstr1.c0     = newthetaI(1);
        myfitstr1.thetaI = newthetaI(2:p);
        newcpar = lsqnonlin(@LSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        newcpar(1) = newthetaI(1);
        
        newmu  = Phimat*newcpar; 

        newdiff  = (Yobs-newmu);  
        newSSE   = -0.5/oldsigmaE*sum(newdiff.^2);
        newprior = -0.5*(newthetaI-oldtheta(1:p))'*oldInvSigma*(newthetaI-oldtheta(1:p));
        num1     = newSSE + newprior;
       
        if log(rand(1))<=(num1-den1)
            oldthetaImat(:,ii) = newthetaI;
            countthetaI(ii)    = countthetaI(ii)+1;
            oldcparmat(:,ii)   = newcpar;

        else
            oldthetaImat(:,ii) = oldthetaI;
            oldcparmat(:,ii)   = oldcpar;

        end
    end
    
    %% Sample theta;

    MuThetaI      = sum(oldthetaImat,2);
    W             = ncohort*oldInvSigma+InvOmega;
    postW         = inv(W);
    temp1         = oldInvSigma*MuThetaI+InvOmega*reshape(eta0,p,1);
    postmu        = W\temp1;
    oldtheta      = (mvnrnd(postmu, postW))';

    %% Sample inverse of SigmaThetaI;

    diffthetaI   = oldthetaImat-oldtheta*ones(1,ncohort);
    newOmega     = diffthetaI*diffthetaI'+S01;
    oldSigma = iwishrnd(newOmega, ncohort+df);
    oldInvSigma = inv(oldSigma);


    %% Sample sigmaE

    oldmumat  = Phimat*oldcparmat;
    diffmat = Yobsmat-oldmumat;
    SSE      = sum(sum(diffmat.^2));

    oldtauE = gamrnd(0.5*ncohort*nobs+a, 1/(0.5*SSE+1/b));
    oldsigmaE = 1/oldtauE;
    
    %% Output.
    
    out1 = input1;
    
    out1.oldtheta     = oldtheta;
    out1.oldthetaImat = oldthetaImat;
    out1.oldcparmat   = oldcparmat;

    out1.oldsigmaE    = oldsigmaE;
    
    out1.oldInvSigma = oldInvSigma;
    
    out1.countthetaI  = countthetaI;

    out1.oldmumat     = oldmumat;

 end

