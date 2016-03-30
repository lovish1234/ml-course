function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

H = sigmoid(X*theta);
% y is m*1 , X is m*(n+1) theta is n+1 * 1, H is m*1
J = - (y'*log(H)+(1-y)'*log(1-H))/(m) + (lambda/(2*m))*(theta(2:end,:)'*theta(2:end,:)); 

 
grad = (X'*(H-y))/(m)+(lambda/m)*theta;
grad(1) = ((X'*(H-y))/m)(1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
