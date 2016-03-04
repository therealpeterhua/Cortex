X = [1, 0; 0, 1; 1, 1; 0, 0]
y = [1; 1; 0; 0]

Theta1 = [0.8647455857465699, 0.17581308569973933, -0.8582788392245729; -0.7299604631858434, 0.8494939087137555, 0.9719484737755888]

Theta2 = [0.3169746186886007, -0.9110837624163778, 0.6480662961076894]

learning_rate = 0.05


m = size(X, 1);
num_labels = size(Theta2, 1);
lambda = 0     %PH:*** change later, if you wanna test

% You need to return the following variables correctly
J = 0;


for iter=1:5
  fprintf('============ New Loop =============')

  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  y_matrix = y;       % here, y is what we're looking for, so...


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

  a1;
  a2;
  a3;

  d3 = a3 - y_matrix;       % 5000 x 10

  % discount bias unit, like in regularization
  shortened_theta2 = Theta2(:, 2:end);
  size(d3 * shortened_theta2);    % 5000 x 26 --> 5000 x 25
  size(sigmoidGradient(z2));      % 5000 x 25
  % d2 and z2 same size, nice!
  d2 = (d3 * shortened_theta2) .* sigmoidGradient(z2);    % d2 is 5000 x 25

  d3;
  d2;

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

  Theta1_grad
  Theta2_grad

  Theta1 -= (learning_rate .* Theta1_grad);
  Theta2 -= (learning_rate .* Theta2_grad);

end
