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

% hypothesis
h = X * theta;

% compute squared error
sqErr = (sum((h - y) .^ 2));

% temporarily remove theta 0 term when calculating regularization term
% once calculated, add theta 0 term back
temp = theta(1);
theta(1) = 0;
costRegTerm = lambda * (sum(theta .^ 2));
theta(1) = temp;

% add regularization term and average cost
regJ = (1/(2*m)) * (sqErr + costRegTerm);

% compute gradient with partial derivative
unRegGrad = (1/m)*(X' * (h - y));
gradRegTerm = ((lambda/m) .* theta);

% set theta 0 gradient to 0
gradRegTerm(1) = 0;

% add regularization term to gradient
regGrad = unRegGrad + gradRegTerm;

% =========================================================================

J = regJ;
grad = regGrad(:);

end
