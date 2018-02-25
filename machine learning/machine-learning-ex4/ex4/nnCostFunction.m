function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
% %        cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% fprintf("Size of Y: %dx%d\n", size(y)); % 5000 x 1 (expected training results)
% fprintf("Size of X: %dx%d\n", size(X)); % 5000 x 401  (5000 training samples and 401 inputs)

% calc forward progageted values (In respect to theta)

% Add bias unit to input layer
a1 = X = [ones(m, 1), X];

% Calc hidden layer (FP)
z2 = X * Theta1'; 
a2 = [ones(m, 1), sigmoid(z2)];
% Add bias unit to hidden layer
z3 = a2 * Theta2';
% Calc output layer 
a3 = hx0 = sigmoid(z3);

% fprintf("Size of hx0: %dx%d\n", size(hx0)); % 5000 x 10 (results (1->10) for each training set)

% See: https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA 
% and  https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/ag_zHUGDEeaXnBKVQldqyw -> Q3


% Note that you should not be regularizing the terms that correspond to the bias
theta1_sum = sum(sum(Theta1(:,2:end) .^ 2));
theta2_sum = sum(sum(Theta2(:,2:end) .^ 2));

reg = lambda / (2*m) * (theta1_sum + theta2_sum);

y_new = zeros(m, num_labels);
% we need to recode the labels as vectors only containg values 0 or 1
% convert to something like [0 0 0 0 0 1] for y = 6 
I = eye(num_labels);
for i = 1:m
 % y_new(i,y(i)) = 1; % direct index version 
 y_new(i,:) = I(:,y(i))'; % "Apply" transposed matrix (Identiy of num labels)
end

% The cost has two parts - the first involves the product of 'y' and log(h)
% Note that 'y' and 'h' are both matrices of size (m x K), and the multiplication in the cost equation is a scalar product for each element in the matrices.
left = -y_new .* log(hx0);
% The second involves the product of (1-y) and log(1-h)
right = (1-y_new) .* log(1 - hx0);
J = (1/m) .* sum(sum(left - right)) + reg;


% -------------------------------------------------------------
% Part 2:: Implement backpropagation
% =========================================================================
% Backpropagate the "error"
% Calc small delta 
Delta3 = hx0 - y;

% fprintf("hx0 %dx%d, theta2: %dx%d delta3: %dx%d\n", size(hx0), size(Theta2), size(Delta3));
% hx0 = 5000 x 10
% theta2 = 10 * 26 
% Delta3 = 5000 x 10 
% fprintf("Sig: %dx%d\n", size(sigmoidGradient(z2)));
% sigmoid = 5000 x 25

% nCostFunction: product: nonconformant arguments (op1 is 5000x26, op2 is 5000x10)
% fprintf("z2: %dx%d\n", size(z2));
% z2 = 5000 x 25 
grad2 = [ones(size(z2,1), 1) sigmoidGradient(z2)];
Delta2 = (Delta3 * Theta2 .* grad2)(:, 2:end);

%fprintf("delta2: %dx%d\n", size(Delta2));
%fprintf("delta3: %dx%d\n", size(Delta3));

% Delta3: 5000 x 10 
% Delta2: 5000 x 25

% Calc accumaltors (Capital delta)
%DELTA1 = ;
%DELTA2 = ;

% TODO ::: FIXME ?

Theta1_grad = X' * Delta2 ./ m;
Theta2_grad = a3' * Delta3 ./ m;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
