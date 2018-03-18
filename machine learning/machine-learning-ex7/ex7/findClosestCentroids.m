function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i = 1:length(X)
    min_dist = 9999999;
    % Theoretically we can write a "per element" function call and search for the "min" idx

    % fprintf("Size X: %dx%d\n", size(X(i, :)));
    % fprintf("Size k: %dx%d\n", size(centroids(1, :)));

    for j = 1:K
        dist = norm(X(i,:) - centroids(j,:), 2) ^ 2;

        % fprintf("k(%d): %d,%d\n", j, centroids(j));
        % fprintf("X(%d): %d,%d\n", i, X(i));
        % fprintf("Dist for X(%d): %d to centroid %d: %d is %.2f\n", i, X(i), j, centroids(j), dist);

        if (dist < min_dist)
            % fprintf("Choosing idx %d for X(%d)\n", j, i);
            min_dist = dist;
            idx(i) = j;
        end
    end

    % fprintf("\n\n");
end


% =============================================================

end

