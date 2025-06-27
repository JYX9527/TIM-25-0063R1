function phi = fit_fun1(param, X)
 
c1 = param(1);
c2 = param(2);
c3 = param(3);
c4 = param(4);
c5 = param(5);
d0 = param(6);
d1 = param(7);
d2 = param(8);
d3 = param(9);
d4 = param(10);
d5 = param(11);
x = X(:,1);
y = X(:,2);
z = X(:,3);
A=c1+c3*x+c5*y;B=1+c2*x+c4*y;C=d1+d3*x+d5*y;D=d0+d2*x+d4*y;
phi = (B - z .* D)./(z .* C - A);
% Z = (1 + c1 * phi + (c2 + c3 * phi).* x + (c4 + c5 * phi).* y)./...
%    (d0 + d1 * phi + (d2 + d3 * phi).* x + (d4 + d5 * phi).* y);
% % ≤‚ ‘¥˙¬Î
% u = X(1,1);
% v = X(1,2);
% phi = X(1,3);
% z = Z(1);
% A_row = [phi, u, phi*u, v, phi*v, -z, -phi*z, -u*z, -phi*u*z, -v*z, -phi*v*z, 1];
% err = A_row * [param;1];

end 