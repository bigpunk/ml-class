function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    learning_factor = alpha/m;
    new_theta = theta;

    for j = 1:length(theta)

      s = 0;
      for i = 1:m
	  x_i_s = X(i, :);
	  hypoth = sum(theta' .* x_i_s);
	  diff = hypoth - y(i);

	  s += diff * X(i, j);
      end

      theta_new(j) = theta(j) - learning_factor * s;
    end
    theta = theta_new';
    


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
