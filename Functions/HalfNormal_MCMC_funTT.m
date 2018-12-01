function out1 = HalfNormal_MCMC_funTT(input1)
% Note: this program is built on Matlab (at least 2014 version) due to
% truncated distributed are used.
% Assuming Student's t distributions on both random-effects and observations.
% A half-normal prior assigned on the standard deviation parameter.
       
        ncohort      = input1.ncohort;
        oldthetaImat = input1.oldthetaImat;
        oldtheta     = input1.oldtheta;
        oldcparmat   = input1.oldcparmat;
        oldUmat      = input1.oldUmat;
        oldWmat      = input1.oldWmat;  
        oldsigmaE    = input1.oldsigmaE; % The standard deviation parameter.
        
        oldscale_sigmaE = input1.oldscale_sigmaE;
 
        oldInvSigma = input1.oldInvSigma;
        oldkappa     = input1.oldkappa;
        oldnu        = input1.oldnu;  
        oldlambda_kappa = input1.oldlambda_kappa;
        oldlambda_nu = input1.oldlambda_nu;

        sig1    = input1.sig1;
        sig2    = input1.sig2;
        sig3    = input1.sig3;
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

        c       = input1.c;  
        d       = input1.d;  
        
        
        D0Qbasismat = input1.D0Qbasismat;
        D1Qbasismat = input1.D1Qbasismat;
        InvOmega       = input1.InvOmega;
         
        countthetaI = input1.countthetaI;
        countkappa  = input1.countkappa;
        countnu     = input1.countnu;
        countsigmaE = input1.countsigmaE;
        
        options1    = input1.options1;
 
        %% Update theta_i
        
    for ii=1:ncohort

        oldthetaI   = oldthetaImat(:,ii);   
        newthetaI   = oldthetaI+normrnd(0,sig1,p,1);

        oldU = oldUmat(ii);
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
        oldSSE   = -0.5*oldU/(oldsigmaE^2)*sum(olddiff.^2);
        oldprior = -0.5*oldWmat(ii)*(oldthetaI-oldtheta(1:p))'*oldInvSigma*(oldthetaI-oldtheta(1:p));
        den1     = oldSSE + oldprior;


        myfitstr1.c0     = newthetaI(1);
        myfitstr1.thetaI = newthetaI(2:p);
        newcpar = lsqnonlin(@LSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        newcpar(1) = newthetaI(1);
        
        newmu  = Phimat*newcpar; 

        newdiff  = (Yobs-newmu);  
        newSSE   = -0.5*oldU/(oldsigmaE^2)*sum(newdiff.^2);
        newprior = -0.5*oldWmat(ii)*(newthetaI-oldtheta(1:p))'*oldInvSigma*(newthetaI-oldtheta(1:p));
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

   
    %% Sample ui
    

    oldmumat = Phimat*oldcparmat;
    diffmat = Yobsmat-oldmumat;
    resids2 = diag(diffmat'*diffmat);
    oldUmat = gamrnd(oldkappa/2+nobs/2,2./(oldkappa+resids2/(oldsigmaE^2)),ncohort,1);
    
    %% Sample wi
    temp1 = oldthetaImat-oldtheta*ones(1,ncohort);
    temp2 = diag(temp1'*oldInvSigma*temp1);
    
    oldWmat = gamrnd(oldnu/2+p/2,2./(oldnu+temp2),ncohort,1);
 
    
    %% Sample kappa
    
    pd = makedist('normal',log(oldkappa),sig2);
    newnormpdf = truncate(pd,log(2.0),inf);
    newkappa = exp(random(newnormpdf,1));
    

    num1 = ncohort*(0.5*newkappa*log(0.5*newkappa)-gammaln(0.5*newkappa))+(0.5*newkappa-1)*sum(log(oldUmat))-0.5*newkappa*sum(oldUmat)-oldlambda_kappa*newkappa;
    den1 = ncohort*(0.5*oldkappa*log(0.5*oldkappa)-gammaln(0.5*oldkappa))+(0.5*oldkappa-1)*sum(log(oldUmat))-0.5*oldkappa*sum(oldUmat)-oldlambda_kappa*oldkappa;
    if log(rand(1))<=(num1-den1)
        oldkappa = newkappa;
        countkappa = countkappa+1;
    end

   %% Sample nu
   
     
    pd = makedist('normal',log(oldnu),sig3);
    newnormpdf = truncate(pd,log(2.0),inf);
    newnu = exp(random(newnormpdf,1));
    
    num1 = ncohort*(0.5*newnu*log(0.5*newnu)-gammaln(0.5*newnu))+(0.5*newnu-1)*sum(log(oldWmat))-0.5*newnu*sum(oldWmat)-oldlambda_nu*newnu;
    den1 = ncohort*(0.5*oldnu*log(0.5*oldnu)-gammaln(0.5*oldnu))+(0.5*oldnu-1)*sum(log(oldWmat))-0.5*oldnu*sum(oldWmat)-oldlambda_nu*oldnu;
    if log(rand(1))<=(num1-den1)
        oldnu = newnu;
        countnu = countnu+1;
    end

    %% Sample lambda_kappa and lambda_nu
        
    pd1 = makedist('gamma',2,1/oldkappa);
    newgampdf1 = truncate(pd1,c,d);
    newlambda_kappa = random(newgampdf1,1);
    pd2 = makedist('gamma',2,1/oldnu);
    newgampdf2 = truncate(pd2,c,d);
    newlambda_nu = random(newgampdf2,1);

    oldlambda_kappa = newlambda_kappa;
    oldlambda_nu = newlambda_nu;
       
    
    %% Sample theta;

    MuThetaI      = oldthetaImat*oldWmat;
    W             = sum(oldWmat)*oldInvSigma+InvOmega;
    postW         = inv(W);
    temp1         = oldInvSigma*MuThetaI+InvOmega*reshape(eta0,p,1);
    postmu        = W\temp1;
    oldtheta      = (mvnrnd(postmu, postW))';

    %% Sample inverse of SigmaThetaI;

    diffthetaI   = oldthetaImat-oldtheta*ones(1,ncohort);
    newOmega     = diffthetaI*diag(oldWmat)*diffthetaI'+S01;
    oldSigma = iwishrnd(newOmega, ncohort+df);
    oldInvSigma = inv(oldSigma);

    
    %% Sample sigmaE
    
    newsigmaE = normrnd(oldsigmaE,sig4);
    diffmat = Yobsmat-oldmumat;
    SSE      = sum(oldUmat'.*sum(diffmat.^2));

    num1 = -(ncohort*nobs)*log(abs(newsigmaE))-SSE/2/(newsigmaE^2)-(newsigmaE^2)/2*oldscale_sigmaE;
    den1 = -(ncohort*nobs)*log(abs(oldsigmaE))-SSE/2/(oldsigmaE^2)-(oldsigmaE^2)/2*oldscale_sigmaE;
    
   % display(num2str(num1-den1));
    if log(rand(1))<=(num1-den1)
        oldsigmaE = newsigmaE;
        countsigmaE = countsigmaE+1;
    end
    
%% Sample scale_sigmaE
    
    oldscale_sigmaE = gamrnd(0.5+0.5, 1/((oldsigmaE^2)/2+2/5));
    
    
    %% Output.
    
    out1 = input1;
    
    out1.oldtheta     = oldtheta;
    out1.oldthetaImat = oldthetaImat;
    out1.oldcparmat   = oldcparmat;
    out1.oldUmat      = oldUmat;
    out1.oldsigmaE    = oldsigmaE;
    out1.oldkappa     = oldkappa;
    out1.oldWmat      = oldWmat;
    out1.oldnu        = oldnu;
    out1.oldlambda_kappa = oldlambda_kappa;
    out1.oldlambda_nu = oldlambda_nu;
    
    out1.oldscale_sigmaE = oldscale_sigmaE;
    
    out1.oldInvSigma = oldInvSigma;
    
    out1.countthetaI  = countthetaI;
    out1.countkappa   = countkappa; 
    out1.countnu      = countnu; 
    out1.countsigmaE  = countsigmaE;
       
    out1.oldmumat     = oldmumat;

 end

