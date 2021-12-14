function [W2,W3,W4,b2,b3,b4,costs] = netbp2(x1,x2,y,Niter)
% NETBP2 Use backpropagation to train a network.  This is an Octave-
% compatible rewrite of NETBP by Higham & Higham (2019) = HH19.
% Derived from:  https://www.maths.ed.ac.uk/~dhigham/algfiles.html
% Sets up a fixed four-layer network (Fig 2.3 in HH19)
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

% Initialize weights and biases
randn('seed',5000);        % reproducable results
W2 = 0.5*randn(2,2); W3 = 0.5*randn(3,2); W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1); b3 = 0.5*randn(3,1); b4 = 0.5*randn(2,1);

% Forward and Back propagate
eta = 0.05;                % learning rate
costs = zeros(Niter,1);    % value of cost function at each iteration
for counter = 1:Niter
    k = randi(10);         % choose a training point at random
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-y(:,k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    costs(counter) = cost(W2,W3,W4,b2,b3,b4);
    if mod(counter,1000) == 0
        fprintf('%9d:  cost = %8.4f\n', counter, costs(counter))
    end
end

    function y = activate(x,W,b)
    % ACTIVATE  Logistic activation function.

    y = 1./(1+exp(-(W*x+b)));
    end

    function costval = cost(W2,W3,W4,b2,b3,b4)
    costvec = zeros(10,1);
    for i = 1:10
        x =[x1(i);x2(i)];
        a2 = activate(x,W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        costvec(i) = norm(y(:,i) - a4,2);
    end
    costval = norm(costvec,2)^2;
    end

end
