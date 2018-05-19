function [Xk, gk, i] = conjugate_gradient(f, Xk, options)
 if (~exist('options','var'))
        options.MaxIter = 50;
        options.epsilon = 0.01;
 end
 
     i = 0;
    [fk,gk] = f(Xk);
    dk = -gk;
   
    %-- Main loop
    while ( (norm(gk)/norm(fk) > options.epsilon) && (i<options.MaxIter) )

        %-- internal iterator
        i = i+1;
       
        Xk= Xk + options.tau*dk;
           
 
        [fk,gkp] = f(Xk);
        bk=(gkp*(gkp-gk)')/(norm(gk)*norm(gk));
        dk = -gkp+bk*dk;
        gk = gkp;
       
        
        %-- display iterative cost value
        S = 'Iteration ';
        f1 = norm(fk);
        fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
        
    end
    fprintf('\n');
    
end

