% EXAMPLE1  Run NETBP2 for the main example in Higham & Higham (2019) = HH19.
% Reproduces Figures 2.1 and 6.2 from HH19.

if exist('Niter')
    Niter
else
    Niter = 1e6;
end

x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

% train
fprintf('training with Niter = %d ...\n', Niter)
[W2,W3,W4,b2,b3,b4,costs] = netbp2(x1,x2,y,Niter);
fprintf('done!\n')

% show data
figure(1)
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
hold on
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
axis([0 1 0 1])
set(gca,'FontWeight','Bold','FontSize',16)
xlabel('x_1'),  ylabel('x_2')
title('data = 10 classified points (x_1,x_2)')

% cost history
figure(2)
semilogy([1:1e3:Niter],costs(1:1e3:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',16)

% FIXME  add result figure 6.2
