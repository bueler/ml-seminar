function classfig(x1,x2,y,Pval)
% CLASSFIG  Show classification figure like Figure 2.4 or Figure 6.2
% in Higham & Higham (2019) = HH19.  Calls GRIDFORWARD.

[X,Y,Aval,Bval] = gridforward(Pval);
Mval = (Aval > Bval);      % compare output neuron values
contourf(X,Y,double(Mval),[0.5 0.5]),  hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(y(1,:)==1),x2(y(1,:)==1),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(y(2,:)==1),x2(y(2,:)==1),'bx','MarkerSize',12,'LineWidth',4)
axis([0 1 0 1]),  set(gca,'FontWeight','Bold','FontSize',16)
hold off
