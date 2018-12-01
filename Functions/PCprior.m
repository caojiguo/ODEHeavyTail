function y = PCprior(nu,lambda)
% Penalized complexity prior.

    %lambda = input1.lambda; % 
    xL = -10;
    xR = 10;
    
    h = 0.01;
    
    quadpts = (xL:h:xR)';
    nquad = length(quadpts);
    quadwts = ones(nquad,1);
    quadwts(2:2:(nquad-1)) = 4;
    quadwts(3:2:(nquad-2)) = 2;
    quadwts = quadwts.*(h/3);

    KLD = (tpdf(quadpts,nu).*log(tpdf(quadpts,nu)))'*quadwts+0.5*log(2*pi)+0.5*nu/(nu-2);
    dv  = sqrt(2*KLD);
    
    dA = 0.5*gamma((nu+1)/2)*psi((nu+1)/2);
    dB = 0.5*gamma(nu/2)*psi(nu/2);
    dC = 0.5*sqrt(pi/(nu-2));
    A  = gamma((nu+1)/2);
    B  = gamma(nu/2);
    C  = sqrt((nu-2)*pi);
    num1 = dA*B*C-A*(dB*C+B*dC);
    den1 = (B*C)^2;
    dConst = num1/den1;
    
    dfdv1 = dConst*exp(-(nu+1)/2*log(1+quadpts.^2/(nu-2)));
    dfdv2 = tpdf(quadpts,nu).*(-1).*(0.5*log(1+quadpts.^2/(nu-2))-(quadpts.^2)./(quadpts.^2+nu-2)-(nu+1)/2/(nu-2));
    dfdv = dfdv1+dfdv2;
    
    dKLDdv = (dfdv.*(log(tpdf(quadpts,nu))-log(normpdf(quadpts,0,1))+1))'*quadwts;    
    
    PCdensity = lambda*exp(-lambda*dv)*abs(dKLDdv/sqrt(2*KLD));
    %PCdensity = lambda*exp(-lambda*dv);

    %y = [KLD; PCdensity];
    y = PCdensity;
end

