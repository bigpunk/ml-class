function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

preds = svmPredict(model, Xval);
best_mean = mean(double(preds ~= yval));

vals = [0.01 0.03 0.1 0.3 1 3 10 30];

for C_temp = vals
  for sigma_temp = vals
      model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, \
							     sigma_temp));
      preds = svmPredict(model, Xval);
      m = mean(double(preds ~= yval));
      if m < best_mean
        C = C_temp;
        sigma = sigma_temp;
        best_mean = m;
      endif
  endfor 
endfor



% =========================================================================

end
