function [W2, W3, W4, b2, b3, b4] = expandp(Pval)
% EXPANDP Expand length 23 vector
%     Pval = [W2(:); W3(:); W4(:); b2; b3; b4;]
% into weight matrices and bias vectors.
W2 = reshape(Pval(1:4),2,2);
W3 = reshape(Pval(5:10),3,2);
W4 = reshape(Pval(11:16),2,3);
b2 = reshape(Pval(17:18),2,1);
b3 = reshape(Pval(19:21),3,1);
b4 = reshape(Pval(22:23),2,1);
end
