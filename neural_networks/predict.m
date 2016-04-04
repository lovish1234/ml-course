function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% 5000 * 401
% add a column of 1's to feature set of X ( for bias )
X = [ ones(m,1) X ]; 

25*401
% get the z1 and apply logistic function
a1 = sigmoid(X*Theta1');

ma1 = size(a1,1);
% add 1 to activations from 1st layer 
a1 = [ ones(ma1,1) a1];

% get the z2 and apply logistic function 
a2 = sigmoid(a1*Theta2');

size(a2);
[y,p] = max(a2,[],2);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
