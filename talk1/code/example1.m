% EXAMPLE1  Run NETBP2 for the main example in Higham & Higham (2019) = HH19.
% Compare figure-generation in NETBPFULL.  Reproduces Figures 2.1, 6.1, and
% 6.2 from HH19.

% user can set global variables Niter and nofigs
if ~exist('Niter')
    Niter = 1e6;
end
nofigs = exist('nofigs');

% the data (see Figure 2.1 in HH19)
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

% train it ... slow
fprintf('training with Niter = %d ...\n', Niter)
[W2,W3,W4,b2,b3,b4,costs] = netbp2(x1,x2,y,Niter,1000);
fprintf('done!\n')
if nofigs,  return,  end  % if user set nofigs to any value then stop

% show data
figure(1)
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
hold on
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
axis([0 1 0 1])
set(gca,'FontWeight','Bold','FontSize',16)
xlabel('x_1'),  ylabel('x_2')
title('compare Figure 2.1')

% show cost history
figure(2)
semilogy(1000*(1:length(costs)),costs,'b-','LineWidth',2)
xlabel('Iteration')
ylabel('Cost function value')
set(gca,'FontWeight','Bold','FontSize',16)
title('compare Figure 6.1 (scale corrected)')

% array of forward-pass results
activate = @(z) 1 ./ (1 + exp(-z));
N = 200;   % vs 500 ... slow
xvals = linspace(0,1,N+1);
yvals = xvals;
[X,Y] = meshgrid(xvals,yvals);
Aval = zeros(size(X));
Bval = Aval;
for k1 = 1:N+1
    for k2 = 1:N+1
        xy = [xvals(k1); yvals(k2)];
        a2 = activate(W2 * xy + b2);
        a3 = activate(W3 * a2 + b3);
        a4 = activate(W4 * a3 + b4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
     end
end

% show classification result
figure(3)
Mval = (Aval > Bval);                  % note 2 output neurons
contourf(X,Y,double(Mval),[0.5 0.5])
colormap([1 1 1; 0.8 0.8 0.8])
hold on
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
axis([0 1 0 1])
set(gca,'FontWeight','Bold','FontSize',16)
title('compare Figure 6.2')
