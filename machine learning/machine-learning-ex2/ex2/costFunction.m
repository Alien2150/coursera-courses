function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% J(0) = 1/m sum [ -y(i) * log(h(x(i)) - (1-y(i))* log(1 - h(x(i)))] 

for i = 1:m
	hypothesis = sigmoid(theta'*X(i));
	J = J + (-y(i) * log(hypothesis) - (1 - y(i)) * log(1 - hypothesis));
end

J = J / m;

% gradient: 
% for i = 1:m 
	%hypothesis = sigmoid(theta'*X(i));
	%grad(i) = sum((hypothesis - y(i))*X(i)) / m;
%end
hypothesis = X * theta;
grad = sum((hypothesis - y) .* X)/m;


for j = 1:length(theta)
	for i = 1:m
		hypothesis = sigmoid(theta'*X(i));
		grad(j) = sum((hypothesis - y(i))*X(j,:)) / m;
	end
end 

% =============================================================

end