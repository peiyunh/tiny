function varargout = unpackColumns(X)

if ndims(X) > 2
    error('Please input a 1-d or 2-d matrix');
end
for i = 1:size(X,2)
    varargout{i} = X(:,i); 
end