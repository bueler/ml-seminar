function [X,Y,Aval,Bval] = gridforward(Pval,m)
% GRIDFORWARD Generate array of forward-pass results.
% Used for plotting classification figure.

if nargin < 2,  m = 201;  end

activate = @(z) 1 ./ (1 + exp(-z));
W2 = reshape(Pval(1:4),2,2);
W3 = reshape(Pval(5:10),3,2);
W4 = reshape(Pval(11:16),2,3);
b2 = reshape(Pval(17:18),2,1);
b3 = reshape(Pval(19:21),3,1);
b4 = reshape(Pval(22:23),2,1);

xvals = linspace(0,1,m);  yvals = xvals;
[X,Y] = meshgrid(xvals,yvals);

Aval = zeros(size(X));  Bval = Aval;
for k1 = 1:m
    for k2 = 1:m
        xy = [xvals(k1); yvals(k2)];
        a2 = activate(W2 * xy + b2);
        a3 = activate(W3 * a2 + b3);
        a4 = activate(W4 * a3 + b4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
     end
end
