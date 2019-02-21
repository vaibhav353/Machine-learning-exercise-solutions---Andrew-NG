function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

x=X(:,2);
for iter = 1:num_iters
    theta_two = theta(2) -(x'*(X*theta - y))*(alpha/m);
    theta_one = theta(1) - (sum(X*theta - y))*(alpha/m);
    theta = [theta_one;theta_two];

    J_history(iter) = computeCost(X, y, theta);

endfor
disp(min(J_history));
end
