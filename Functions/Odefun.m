function dy = Odefun(t,y,input)
   
    a = input.a;
    b = input.b;
    
    dy  = -a*y+b;
    
end