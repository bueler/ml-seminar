function y = activate(z)
% ACTIVATE  Logistic activation function, applied entry-wise.
% See equation (2.1) in HH19.
y = 1 ./ (1 + exp(-z));
end
