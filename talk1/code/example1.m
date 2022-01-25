function Pval = example1(optmethod,Niter,figs,repeatable)
% EXAMPLE1  Run NETBP2 or NETOPT for the main example in
% Higham & Higham (2019) = HH19.  (Compare NETBPFULL from HH19.)
% Default solver uses NETBP2 and reproduces Figures 6.1, 6.2 from
% HH19.  Calls CLASSFIG to generate classification figure.
% Usage:
%   Pval = example1(optmethod,Niter,figs,repeatable)
% inputs:
%   optmethod   'sgbp' = stochastic gradient back-propagation [default]
%               'nm'   = Nelder-Mead
%   Niter       number of training iterations (for sgbp)
%               or cost-function evaluations (for nm)
%   figs        if true [default] then show figures
%   repeatable  if true [default] then set seed on random number gen
% outputs:
%   Pval        parameter values after training; see EXPANDP
% Examples:
%   >> example1;                                 % default behavior
%   >> Pval = example1('sgbp',1e6,true,true);    % also defaults
%   >> Pval = example1('nm',10000,true,false);   % sometimes finds a global min

% settings
if nargin < 4,  repeatable = true;  end
if nargin < 3,  figs = true;  end
if nargin < 1,  optmethod = 'sgbp';  end
if nargin < 2
    if strcmp(optmethod,'sgbp'),  Niter = 1e6;
    else,                         Niter = 1e3;  end
end

% the data (see Figure 2.1 in HH19)
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

% initialize parameters to random values
if repeatable
    % set seed for reproducable results
    isOctave = (exist('OCTAVE_VERSION', 'builtin') ~= 0);
    if isOctave,  randn('seed',5000);  rand('seed',5000);
    else,  rng(5000);  end
end
Pzero = 0.5 * randn(23,1);

% train it
if strcmp(optmethod,'sgbp')
    fprintf('training using SG and BP with Niter = %d ...\n', Niter)
    [Pval,costs] = netbp2(x1,x2,y,Pzero,Niter,1000);
elseif strcmp(optmethod,'nm')
    fprintf('training using Nelder-Mead, at most %d cost evals ...\n', Niter)
    [Pval,finalC] = netopt(x1,x2,y,Pzero,Niter);
else,  error('unsupported value for optimize'),  end
fprintf('done!\n')
if ~figs,  return,  end

% visualize classification result; compare Figure 6.2
figure(1)
classfig(x1,x2,y,Pval);

% show cost history if available; compare Figure 6.1
if strcmp(optmethod,'sgbp')
    figure(2)
    semilogy(1000*(1:length(costs)),costs,'b-','LineWidth',2)
    xlabel('Iteration'),  ylabel('Cost function value')
    set(gca,'FontWeight','Bold','FontSize',16)
end
