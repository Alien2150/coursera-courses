function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
fprintf("Theta: %dx%d, x: %dx%d, y: %dx%d\n", size(theta), size(X), size(y));

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	old_theta = theta;
	
	for j = 1:length(theta)
		hypothesis = X * old_theta;
		theta(j) = theta(j) - (alpha/m) * sum((hypothesis - y).* X(:,j));
	end 
	
	fprintf("Iteration %d: with theta: (%0.3f, %0.3f) = %0.3f\n", iter, theta(1), theta(2), computeCost(X, y, theta));
    % ============================================================

    % Save the cost J in every iteration
	costs = computeCost(X, y, theta);
    J_history(iter) = costs;
	
end

theta

end
