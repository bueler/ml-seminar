% EXAMPLE1  Run NETBP2 or NETOPT for the main example in
% Higham & Higham (2019) = HH19.  Compare NETBPFULL.  Defaults
% to using NETBP2 and reproduces Figures 6.1, 6.2 from HH19.
% Calls GRIDFORWARD to generate classification figure.

% user can set global variables: nofigs, optimize, Niter
nofigs = exist('nofigs');
if ~exist('optimize')
    optimize = 'sgbp';
end

% the data (see Figure 2.1 in HH19)
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

% train it
if strcmp(optimize,'sgbp')
    if ~exist('Niter'),  Niter = 1e6;  end
    fprintf('training using SG and BP with Niter = %d ...\n', Niter)
    [Pval,costs] = netbp2(x1,x2,y,Niter,1000);
elseif strcmp(optimize,'nm')
    if ~exist('Niter'),  Niter = 1e3;  end
    fprintf('training using Nelder-Mead, at most %d cost evals ...\n', Niter)
    [Pval,finalC] = netopt(x1,x2,y,Niter);
else
    error('unsupported value for optimize')
end
fprintf('done!\n')
if nofigs,  return,  end  % if user set nofigs to any value then stop

% visualize classification result
[X,Y,Aval,Bval] = gridforward(Pval);
Mval = (Aval > Bval);      % compare output neuron values
figure(1)
contourf(X,Y,double(Mval),[0.5 0.5]),  hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
axis([0 1 0 1])
set(gca,'FontWeight','Bold','FontSize',16)
title('compare Figure 6.2')

% show cost history if available
if strcmp(optimize,'sgbp')
    figure(2)
    semilogy(1000*(1:length(costs)),costs,'b-','LineWidth',2)
    xlabel('Iteration'),  ylabel('Cost function value')
    set(gca,'FontWeight','Bold','FontSize',16)
    title('compare Figure 6.1 (scale corrected)')
end
