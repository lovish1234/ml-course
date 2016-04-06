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

%Theta_1 is hidden layer size * input_layer_size+1
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

%Theta_2 is num_labels * hidden_layer_size+1
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

%Theta_1 is hidden layer size * input_layer_size+1
%Theta_2 is num_labels * hidden_layer_size+1
% X is number_of_examples*(input_layer_size+1)
%z_2 is number_of_examples * hidden_layer_size
%a_2 is number_of_examples * hidden_layer_size
%z_3 is num of examples * num_of_labels
%a_3 is num of examples * num_of_labels

%we should add the bias too here 
X = [ ones(m,1) X ];

a_1 = X;

z_2 = X*(Theta1)';
a_2 =[ ones(m,1) sigmoid(z_2) ];

%we should add the bias too here 
z_3 = a_2*(Theta2');
a_3 = sigmoid(z_3);

yy = zeros(size(y),num_labels);
for i=1:m
yy(i,y(i))=1;
end
%cost function 
% convert both y and z_3 into number_of_examples* number_of_labels with 0 and 1 
% when you calculate error difference, you dont take the max and difference

% the method below is vectorized but quiet inefficient
%J=-(log(a_3)*yy'+log(1-a_3)*(1-yy'));
%J=sum(diag(J))/m;

% below is a better vectorized way of determining cost function 
J=-(sum(sum(log(a_3).*yy,2))+sum(sum(log(1-a_3).*(1-yy')',2)))/(m);

% regularization
J+= (lambda/(2*m))* ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );


% delta_3 is number_of_examples * number_of_labels
% Theta2 is number_of_labels* hidden_layer_size+1
% sigmoidGradient(z_2) is number_of_examples * hidden_layer_size+1
% delta_2 is number_of_examples * hidden_layer_size
delta_3 = a_3 - yy;

% remove the delta to bias 
delta_2 = (delta_3*Theta2).*([ ones(m,1) sigmoidGradient(z_2) ]); 
delta_2 = delta_2(:,2:end);

size(delta_2);
size(a_1);

size(delta_3);
size(a_2);

%a_2 is number_of_examples * hidden_layer_size


Theta1_grad = (delta_2'*a_1)/(m)+((lambda)/(m))*[ zeros(size(Theta1,1),1) Theta1(:,2:end)] ;
Theta2_grad = (delta_3'*a_2)/(m)+((lambda)/(m))*[ zeros(size(Theta2,1),1) Theta2(:,2:end)] ;

%Delta_3 is 1 * number_labels
%Delta_2 is 1 * hidden_layer_size 

grad = [ Theta1_grad(:) ; Theta2_grad(:) ];

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

% have to unroll Theta1_grad and Theta2_grad to get grad 


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
