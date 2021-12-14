function [W2,W3,W4,b2,b3,b4,costs] = netbp2(x1,x2,y,Niter)
% NETBP2 Use backpropagation to train a network.  This is an Octave-
% compatible rewrite of NETBP by Higham & Higham (2019) = HH19.
% Derived from:  https://www.maths.ed.ac.uk/~dhigham/algfiles.html
% Sets up a fixed-topology four-layer network (Fig 2.3 in HH19)
% Usage:
%   [W2,W3,W4,b2,b3,b4,costs] = netbp2(x1,x2,y,Niter)
% where:
%   x1, x2      input data
%   y           output data
%   Niter       number of stochastic gradient iterations
%   W2, W3, W4  weight matrices (after training)
%   b2, b3, b4  bias vectors (after training)
%   costs       cost (objective) value history
% Example:
%   >> example1

% Check data
x1 = x1(:);  x2 = x2(:);   % force into columns
if length(x1) ~= length(x2)
    error('input data x1,x2 must be vectors of same length m')
end
N = length(x1);            % number of pieces of data
if any(size(y) ~= [2,N])
    error('output data y must be 2 x m where m = length(x1)')
end

% Initialize weights and biases:  fixed network topology!
randn('seed',5000);        % for reproducable results
W2 = 0.5*randn(2,2); W3 = 0.5*randn(3,2); W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1); b3 = 0.5*randn(3,1); b4 = 0.5*randn(2,1);

% Forward pass and back propagate
eta = 0.05;                % learning rate
costs = zeros(Niter,1);    % value of cost function at each iteration
for counter = 1:Niter
    k = randi(N);          % choose a training point at random
    % Forward pass
    a2 = activate([x1(k); x2(k)],W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-y(:,k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*[x1(k) x2(k)];
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    costs(counter) = cost(W2,W3,W4,b2,b3,b4);
    if mod(counter,1000) == 0
        fprintf('%9d:  cost = %.3e\n', counter, costs(counter))
    end
end

    function y = activate(x,W,b)
    % ACTIVATE  Logistic activation function.
    y = 1./(1+exp(-(W*x+b)));
    end

    function costval = cost(W2,W3,W4,b2,b3,b4)
    % COST  Evaluate cost functional.
    costval = 0;
    for i = 1:N
        a2 = activate([x1(i); x2(i)],W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        costval = costval + norm(y(:,i) - a4,2)^2;
    end
    costval = costval / (2 * N);
    end

end
