function y = dactivate(a)
% DACTIVATE  da/dx computed as a function of a(x).  See
% equation (2.2) in HH19.
y = a .* (1 - a);
end
