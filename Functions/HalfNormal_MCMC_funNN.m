function out1 = HalfNormal_MCMC_funNN(input1)
% Note: this program is built on Matlab (at least 2014 version) due to
% truncated distributed are used.
% Assuming Normal distributions on both random-effects and observations.
% half-normal prior assigned on the standard deviation parameter.
       
        ncohort      = input1.ncohort;
        oldthetaImat = input1.oldthetaImat;
        oldtheta     = input1.oldtheta;
        oldcparmat   = input1.oldcparmat;

        oldsigmaE    = input1.oldsigmaE;
        oldscale_sigmaE = input1.oldscale_sigmaE;
 
        oldInvSigma = input1.oldInvSigma;

        sig1    = input1.sig1;
        sig4    = input1.sig4;
        
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
        InvOmega       = input1.InvOmega;
         
        countthetaI = input1.countthetaI;
        countsigmaE   = input1.countsigmaE;
        
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
        oldSSE   = -0.5/(oldsigmaE^2)*sum(olddiff.^2);
        oldprior = -0.5*(oldthetaI-oldtheta(1:p))'*oldInvSigma*(oldthetaI-oldtheta(1:p));
        den1     = oldSSE + oldprior;


        myfitstr1.c0     = newthetaI(1);
        myfitstr1.thetaI = newthetaI(2:p);
        newcpar = lsqnonlin(@LSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        newcpar(1) = newthetaI(1);
        
        newmu  = Phimat*newcpar; 

        newdiff  = (Yobs-newmu);  
        newSSE   = -0.5/(oldsigmaE^2)*sum(newdiff.^2);
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
    
    newsigmaE = normrnd(oldsigmaE,sig4);
    
    oldmumat  = Phimat*oldcparmat;
    diffmumat = Yobsmat-oldmumat;
    SSE      = sum(sum(diffmumat.^2));

    num1 = -(ncohort*nobs)*log(abs(newsigmaE))-SSE/2/(newsigmaE^2)-(newsigmaE^2)/2*oldscale_sigmaE;
    den1 = -(ncohort*nobs)*log(abs(oldsigmaE))-SSE/2/(oldsigmaE^2)-(oldsigmaE^2)/2*oldscale_sigmaE;
    
   % display(num2str(num1-den1));
    if log(rand(1))<=(num1-den1)
        oldsigmaE = newsigmaE;
        countsigmaE = countsigmaE+1;
    end
    
%% Sample scale_sigmaE
    
    %oldscale_tauE = gamrnd(0.5+0.1, 1/(exp(-2*oldlntauE)/2+0.1));
    oldscale_sigmaE = gamrnd(0.5+0.5, 1/((oldsigmaE^2)/2+2/5));
    
     
    %% Output.
    
    out1 = input1;
    
    out1.oldtheta     = oldtheta;
    out1.oldthetaImat = oldthetaImat;
    out1.oldcparmat   = oldcparmat;

    out1.oldsigmaE    = oldsigmaE;
    
    out1.oldscale_sigmaE = oldscale_sigmaE;
    
    
    out1.oldInvSigma = oldInvSigma;
    
    out1.countthetaI  = countthetaI;
    out1.countsigmaE  = countsigmaE; 
    
           
    out1.oldmumat     = oldmumat;

 end

