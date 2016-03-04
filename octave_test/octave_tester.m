X = [3, 5]
y = [1]

Theta1 = [1, 3, 5; -2, 3, -1]

Theta2 = [1.5, 2, -4]


m = size(X, 1);
num_labels = size(Theta2, 1);
lambda = 1      %PH:*** change later, if you wanna test

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ================= COST FN =======================

size(y);       % 5000 x 1
size(X);       % 5000 x 400
size(Theta1);  % 25 x 401
size(Theta2);  % 10 x 26

% The below is identical to the predict.m from last week!
%PH: AGAIN, this, returns a row of feature probabilities for each row of x. That is, it's a 5000 x 10 matrix, where each of the column of 10 represents the probability of being that number...

second_layer = sigmoid([ones(m, 1) X] * Theta1');
predicted_y = sigmoid([ones(m, 1) second_layer] * Theta2');

% we need to convert actual y into a matrix, not a vector, so we can compare it vs. our 10-vector neural net output (the inner loop on the cost function)
y_matrix = eye(num_labels)(y,:);

errors = -y_matrix .* log(predicted_y) - (1 - y_matrix) .* log(1 - predicted_y);
J = sum(sum(errors)) * (1 / m);     % 1 sum sums cols, need to sum whole thing


theta1_squared = Theta1 .^ 2;
theta2_squared = Theta2 .^ 2;

% REM you do prev_node * theta'. The first el of a row in theta will correspond to the weight of the BIAS node of the prev_layer
theta1_squared(:, 1) = 0;        % get rid of bias regularization
theta2_squared(:, 1) = 0;

% again, need to sum 2-dimensional matrix
regularized_term = sum(sum(theta1_squared)) + sum(sum(theta2_squared));
J += (lambda / (2 * m)) * regularized_term;
J;



% ================= GRADIENT DESCENT =======================


a1 = X;       % 5000 x 400
m1 = size(a1, 1);     % pad a1
a1 = [ones(m1, 1) a1];    % 5000 x 401

z2 = a1 * Theta1';
a2 = sigmoid(z2);
size(a2);      % 5000 x 25
m2 = size(a2, 1);       % pad a2
a2 = [ones(m2, 1) a2];      % 5000 x 26

z3 = a2 * Theta2';
a3 = sigmoid(z3);    % 5000 x 10

a1
a2
a3

d3 = a3 - y_matrix;       % 5000 x 10

% discount bias unit, like in regularization
shortened_theta2 = Theta2(:, 2:end);
size(d3 * shortened_theta2);    % 5000 x 26 --> 5000 x 25
size(sigmoidGradient(z2));      % 5000 x 25
% d2 and z2 same size, nice!
d2 = (d3 * shortened_theta2) .* sigmoidGradient(z2);    % d2 is 5000 x 25

Delta1 = d2' * a1;      % 25 x 401

Delta2 = d3' * a2;      % 10 x 26

Theta1_non_bias = Theta1;
Theta1_non_bias(:, 1) = 0;
Theta2_non_bias = Theta2;
Theta2_non_bias(:, 1) = 0;

Delta1_reg_term = (lambda / m) .* Theta1_non_bias;
Delta2_reg_term = (lambda / m) .* Theta2_non_bias;

Theta1_grad = Delta1 ./ m + Delta1_reg_term;
Theta2_grad = Delta2 ./ m + Delta2_reg_term;
