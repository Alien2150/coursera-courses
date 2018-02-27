function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% Calc costs 

t_new = theta;
t_new(1) = 0;

% hypothesis 
hx0 = X * theta;
diff = hx0 - y; % 12x1
% costs 
reg = ((lambda/(2*m)) * sum(t_new .^ 2));
J = 1/(2*m) * sum((diff) .^ 2) + reg;

% grad:
if (lambda == 0) 
    reg = 0;
else 
    reg = ((lambda / m) * t_new);
end 
grad = (X' * diff)/m + reg; % right term will be 0 for t_new(1)
% =========================================================================

grad = grad(:);

end
