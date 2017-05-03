function Y = tile3DHeat(X)
if isa(X, 'gpuArray')
    X = gather(X);
end

[I,J] = factorize(size(X,3));

%
Y = zeros(size(X,1)*I, size(X,2)*J);
for i = 1:I
    for j = 1:J
        Y(size(X,1)*(i-1)+1:size(X,1)*i, ...
          size(X,2)*(j-1)+1:size(X,2)*j) = X(:,:,(i-1)*J+j);
    end
end

