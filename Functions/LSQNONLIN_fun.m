function f = LSQNONLIN_fun(cpar, myfitstr1)

        D0Qbasismat  = myfitstr1.D0Qbasismat;
        D1Qbasismat  = myfitstr1.D1Qbasismat;
        quadwts      = myfitstr1.quadwts;

        c0     = myfitstr1.c0;
        thetaI = myfitstr1.thetaI;        
        a  = thetaI(1);
        b  = thetaI(2);
        
        cpar(1) = c0;
        

        Xval     = D0Qbasismat*cpar;       
        DXval    = D1Qbasismat*cpar;
               
        diffPEN  = DXval - (-a*Xval+b);
        f1       = sqrt(quadwts);
        f        = f1.*diffPEN;
        
 end

