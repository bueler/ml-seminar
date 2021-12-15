function [Pval,finalcost] = netopt(x1,x2,y,Niter,repeatable)
% NETOPT FIXME
%FIXME want to compare [finalP, finalerr] = lsqnonlin(@neterr,Pzero);

% Check data
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if nargin < 5,  repeatable = false;  end
if nargin < 4,  error('at least 4 arguments required'),  end
x1 = x1(:);  x2 = x2(:);   % force into columns
if length(x1) ~= length(x2)
    error('input data x1,x2 must be vectors of same length N')
end
N = length(x1);            % number of pieces of data
if any(size(y) ~= [2,N])
    error('output data y must be 2 x N where N = length(x1)')
end

% random initial state
if repeatable
    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    if isOctave
        randn('seed',5000);
    else
        rng(5000);
    end
end
Pzero = 0.5 * randn(23,1);

% solve optimization problem
f = @(Pval) norm(neterr(Pval),2)^2 / (2*N);
fprintf('initial cost = %.5f\n',f(Pzero))
opts = optimset('MaxFunEvals',Niter);
[Pval, finalcost] = fminsearch(f, Pzero, opts);
fprintf('final cost   = %.5f\n',finalcost)

% visualize classification result
[X,Y,Aval,Bval] = gridforward(Pval);
figure
Mval = (Aval > Bval);                  % note 2 output neurons
contourf(X,Y,double(Mval),[0.5 0.5])
colormap([1 1 1; 0.8 0.8 0.8])
hold on
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
axis([0 1 0 1])
set(gca,'FontWeight','Bold','FontSize',16)
title('compare Figure 6.2')

    function r = neterr(Pval)
    % NETERR Evaluate the norm residuals from applying the network
    % parameters in Pval to the data x1,x2 and outputs y:
    %    r = [ |y^{1} - F(x^{1})|_2, ..., |y^{N} - F(x^{N})|_2 ]
    % Here Pval is a length 23 vector:
    %    Pval = [W2(:); W3(:); W4(:); b2; b3; b4;]
    % See equations (2.4)-(2.6) in Higham & Higham (2019) = HH19.
    % See NLSRUN at https://www.maths.ed.ac.uk/~dhigham/algfiles.html
    W2 = reshape(Pval(1:4),2,2);
    W3 = reshape(Pval(5:10),3,2);
    W4 = reshape(Pval(11:16),2,3);
    b2 = reshape(Pval(17:18),2,1);
    b3 = reshape(Pval(19:21),3,1);
    b4 = reshape(Pval(22:23),2,1);
    r = zeros(N,1); 
    for i = 1:N
       a2 = activate(W2 * [x1(i);x2(i)] + b2);
       a3 = activate(W3 * a2 + b3);
       a4 = activate(W4 * a3 + b4);
       r(i) = norm(y(:,i) - a4,2);
    end
    end

    function y = activate(z)
    % ACTIVATE  Logistic activation function, applied entry-wise.
    % See equation (2.1) in HH19.
    y = 1 ./ (1 + exp(-z));
    end
end