function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% fprintf("K %d, m %d, n %d\n", K, m, n);
% K = 3 
% m = 300 
% n = 2

for i = 1:K
    % todo :: filter every X(j) where idx(j) == i
    sum = 0;
    length = 0;

    % Filter assigned entries (vectorized possible ?)
    for j = 1:m
        if (idx(j) == i)
            % sum them 
            sum = sum + X(j, :);
            % and calculate mean
            length = length + 1;
        end
    end

    % fprintf("K=%d => Sum: %dx%d\n", i, sum);

    % Calc new centroids
    centroids(i, :) = sum/length;
end


% =============================================================


end

