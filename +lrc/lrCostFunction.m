 

function [J, grad] = lrCostFunction(phi, X, y)

    %-- LRCOSTFUNCTION Compute cost and gradient for logistic regression
    %--   J = LRCOSTFUNCTION(phi, X, y, lambda) computes the cost of using
    %--   theta as the parameter for logistic regression and the
    %--   gradient of the cost w.r.t. to the parameters. 
    
    
    [m,n] = size(X);
    %-- Initialization of energy value J
    J = 0;
    
    
    % ====================== YOUR CODE HERE =========================
    h = lrc.sigmoid(X*phi');
    J = -(1/m) * ( y'*log(h) + (1-y)'*log(1-h) );
    %-- Initialization of gradient vector
    [m,n] = size(X);
    grad = zeros(1,n);
  
    h = lrc.sigmoid(X*phi');
    grad = (1/m)*((h-y)'*X);
  

    
   
    % YOU SHOULD COMPUTE 
    %   - THE VALUE OF THE ENERGY FUNCTION J
    %   - THE GRADIENT VECTOR OF THE ENERGY FUNCTION J
    % ===============================================================
    


end
