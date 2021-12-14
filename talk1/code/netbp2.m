function [W2,W3,W4,b2,b3,b4,costs] = netbp2(x1,x2,y,Niter)
% NETBP2 Use stochastic gradients and backpropagation to train a network.
% This is an Octave-compatible rewrite of NETBP by Higham & Higham (2019)
% = HH19.  See:  https://www.maths.ed.ac.uk/~dhigham/algfiles.html
% Sets up a fixed-topology four-layer network (Fig 2.3 in HH19).
% See EXAMPLE1 for an example.
% Usage:
%   [W2,W3,W4,b2,b3,b4,costs] = netbp2(x1,x2,y,Niter)
% where:
%   x1, x2      input data (size 1 x N)
%   y           output data (size 2 x N)
%   Niter       number of stochastic gradient iterations
%   W2, W3, W4  weight matrices (after training)
%   b2, b3, b4  bias vectors (after training)
%   costs       cost (objective) value history, every 100 iters

% Check data
x1 = x1(:);  x2 = x2(:);   % force into columns
if length(x1) ~= length(x2)
    error('input data x1,x2 must be vectors of same length N')
end
N = length(x1);            % number of pieces of data
if any(size(y) ~= [2,N])
    error('output data y must be 2 x N where N = length(x1)')
end

% Initialize weights and biases:  fixed network topology!
randn('seed',5000);        % for reproducable results
W2 = 0.5*randn(2,2); W3 = 0.5*randn(3,2); W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1); b3 = 0.5*randn(3,1); b4 = 0.5*randn(2,1);

% Do stochastic gradient iterations
eta = 0.05;                % learning rate ... a magic number
costs = zeros(Niter/100,1);  % value of cost function every 100 iterations
counter = 1;
for j = 1:Niter
    k = randi(N);          % choose a training point at random
    % Forward pass; equation (3.2) in HH19
    a2 = activate(W2 * [x1(k); x2(k)] + b2);
    a3 = activate(W3 * a2 + b3);
    a4 = activate(W4 * a3 + b4);
    % Back-propagate; equations (5.5), (5.6)
    delta4 = dactivate(a4) .* (a4 - y(:,k));
    delta3 = dactivate(a3) .* (W4' * delta4);
    delta2 = dactivate(a2) .* (W3' * delta3);
    % Gradient descent step; equation (5.8) for W and (5.7) for b
    W2 = W2 - eta * delta2 * [x1(k) x2(k)];
    W3 = W3 - eta * delta3 * a2';
    W4 = W4 - eta * delta4 * a3';
    b2 = b2 - eta * delta2;
    b3 = b3 - eta * delta3;
    b4 = b4 - eta * delta4;
    % Monitor progress; only evaluate cost every 1000 iterations
    if mod(j,100) == 0
        costs(counter) = cost(W2,W3,W4,b2,b3,b4);
        fprintf('%9d:  cost = %.3e\n', j, costs(counter))
        counter = counter + 1;
    end
end

    function y = activate(z)
    % ACTIVATE  Logistic activation function, applied entry-wise.
    % See equation (2.1) in HH19.
    y = 1 ./ (1 + exp(-z));
    end

    function y = dactivate(a)
    % DACTIVATE  da/dx computed as a function of a(x).  See
    % equation (2.2) in HH19.
    y = a .* (1 - a);
    end

    function costval = cost(W2,W3,W4,b2,b3,b4)
    % COST  Evaluate cost functional.  Equation (3.3) in HH19.
    costval = 0;
    for i = 1:N
        a2 = activate(W2 * [x1(i); x2(i)] + b2);
        a3 = activate(W3 * a2 + b3);
        a4 = activate(W4 * a3 + b4);
        costval = costval + norm(y(:,i) - a4,2)^2;
    end
    costval = costval / (2 * N);
    end

end
