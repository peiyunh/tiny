function [I,J] = factorize(N)
I = [];
J = [];
for i = 1:N
    if rem(N, i) == 0
        I(end+1) = i;
        J(end+1) = N / i ; 
    end
end

% 
d = abs(I - J);
[~,v] = min(d);
%
I = I(v);
J = J(v);
