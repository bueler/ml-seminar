function [Pval,finalcost] = netopt(x1,x2,y,Pzero,Niter)
% NETOPT Use Nelder-Mead derivative-free optimization (FMINSEARCH)
% to train a network.  Sets up a fixed four-layer network (Fig 2.3
% in Higham & Higham (2019) = HH19).  Calls ACTIVATE and EXPANDP.
% See EXAMPLE1 for an example.
% Usage:
%   [Pval,finalcost] = netopt(x1,x2,y,Pzero,Niter,monitorN)
% inputs:
%   x1, x2      input data (size 1 x N)
%   y           output data (size 2 x N)
%   Pzero       initial values for 23 parameters; see EXPANDP
%   Niter       number of stochastic gradient iterations
%   monitorN    [optional] print and save cost value every monitorN iters
% outputs:
%   Pval        parameter values after training, a column vector built
%               from weight matrices and bias vectors;
%               = [W2(:); W3(:); W4(:); b2; b3; b4]
%   finalcosts  final cost (objective) value

% Check inputs
if nargin < 5,  error('at least 5 arguments required'),  end
x1 = x1(:);  x2 = x2(:);   % force into columns
if length(x1) ~= length(x2)
    error('input data x1,x2 must be vectors of same length N')
end
N = length(x1);            % number of pieces of data
if any(size(y) ~= [2,N])
    error('output data y must be 2 x N where N = length(x1)')
end

% solve optimization problem
f = @(Pval) norm(neterr(Pval),2)^2 / (2*N);
fprintf('initial cost = %.5f\n',f(Pzero))

% FIXME regularized version with magic scaling:
% f = @(Pval) norm(neterr(Pval),2)^2 / (2*N) + norm(Pval)^2 / N^4;

% FIXME compare quasi-newton method, but f() should also return gradient:
% [Pval, finalcost] = fminunc(f, Pzero, opts);

% FIXME want to compare Levenberg-Marquart:
% [Pval, cost] = lsqnonlin(@neterr,Pzero);

opts = optimset('MaxFunEvals',Niter);
[Pval, finalcost] = fminsearch(f, Pzero, opts);
fprintf('final cost   = %.5f\n',finalcost)

    function r = neterr(Pval)
    % NETERR Evaluate the norm residuals from applying the network
    % parameters in Pval to the data x1,x2 and outputs y:
    %     r = [ |y^{1} - F(x^{1})|_2, ..., |y^{N} - F(x^{N})|_2 ]
    % See equations (2.4)-(2.6) in Higham & Higham (2019) = HH19.
    % See NLSRUN at https://www.maths.ed.ac.uk/~dhigham/algfiles.html
    [W2, W3, W4, b2, b3, b4] = expandp(Pval);
    r = zeros(N,1); 
    for i = 1:N
       a2 = activate(W2 * [x1(i);x2(i)] + b2);
       a3 = activate(W3 * a2 + b3);
       a4 = activate(W4 * a3 + b4);
       r(i) = norm(y(:,i) - a4,2);
    end
    end
end
