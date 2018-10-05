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
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

A_1 = [ones(size(X, 1), 1) X];
size(A_1)

A_2 = sigmoid(A_1 * Theta1');
A_2 = [ones(size(A_2, 1), 1) A_2];
size(A_2)

A_3 = sigmoid(A_2 * Theta2');
size(A_3)
Y = zeros(num_labels,size(y)(1));
for i = 1:size(y)(1)
  Y(y(i), i) = 1;
end

% J = (log(A_3) * -Y) - (log(1-A_3) * (1-Y));
% size(J)
% J = (1/m) * sum(diag(J));

h = A_3';
J = (1/m) * sum(sum((-Y .* log(h)) - ((1-Y) .* log(1- h))));

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
%delta_2_fin
for i=1:m
  a_1 = X(i, :);
  a_1 = [1 a_1];
  a_2 = sigmoid(a_1 * Theta1');
  a_2 = [1 a_2];
  a_3 = sigmoid(a_2 * Theta2');
  h = a_3';


  delta_3 = h - Y(:, i);
  delta_2 = (Theta2' * delta_3)(2:end,:) .* (sigmoidGradient(a_1 * Theta1'))';
  Theta1_grad = Theta1_grad + delta_2*a_1;
  Theta2_grad = Theta2_grad + delta_3*a_2;
end

Theta1_grad(:, 1) = (1/m) * Theta1_grad(:, 1);
Theta1_grad(:, 2:end) = (1/m) * Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 1) = (1/m) * Theta2_grad(:, 1);
Theta2_grad(:, 2:end) = (1/m) * Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));

% delta_3 = A_3' - Y;
% size(delta_3)
% size(Theta1')
% g_prime = sigmoidGradient(A_1 * Theta1');
% size(g_prime)
% test = delta_3 .* g_prime;
% delta_2 = Theta2' * test;
% delta_2_fin = zeros(size(Theta2)(1), size(Theta2)(2));
% delta_2_fin = delta_2_fin + delta_3 * A_2;
% delta_1_fin = zeros(size(Theta1)(1), size(Theta1)(2));
% delta_1_fin = delta_1_fin + delta_2(2:end, :) * A_1;
%
% Theta1_grad = (1/m) * delta_1_fin;
% Theta2_grad = (1/m) * delta_2_fin;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
theta1_reg = Theta1(:, 2:end);
theta2_reg = Theta2(:, 2:end);
reg = sum(sum( theta1_reg .* theta1_reg )) + sum(sum( theta2_reg .* theta2_reg ));
J = J + reg * ( lambda / (2*m) ) ;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
