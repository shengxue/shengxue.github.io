---
layout: post
title: Exercise 1- Linear Regression
description: Exercise 1 of the standord Corsera course Machine learning
modified: 2016-10-17
tags: [Machine learning, Stanford, Assignment]
image:
  feature: norway-483185_1920.jpg
  credit: pixabay
  creditlink: https://pixabay.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---
# 1. Sublime Text3 Octave build system
```json
{
    "cmd": ["octave-gui", "$file"],
    "shell": true    // to show plots
}
```

# 2. Linear regression with one variable

## plotdata.m
```m
function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

figure; % open a new figure window


plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');



% ============================================================

end
```

## Cost function

### Equation
The objective of linear regression is to minimize the cost function
\\[J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2\\]
where the hypothesis \\(h_\theta(x)\\) is given by the linear model
\\[h_\theta(x) = \theta^T x = \theta_0 + \theta_1\\]

### Implementation
```m
% non-vectorized version.
J = 0;
for i=1:m
  dif = X(i, :)*theta-y(i);
  J = J + dif*dif;
endfor
J = J / (2*m);


% vectorized version.
dif = X*theta-y;
J = (dif'*dif)/(2*m);
```

## Gradient decent

### Equation
\\[\theta_j = \theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}\\]

### Implementation

* Version (1)

```m
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    theta_prev = theta;
    p = length(theta);

    for j = 1:p

        sum = 0;
        for i = 1:m
            sum = sum + (X(i,:)*theta_prev - y(i))*X(i,j);
        end

        derive = sum/m;
        theta(j) = theta(j) - alpha*derive;
    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
```

* version (2)

``` m
    theta_prev = theta;
    p = length(theta);

    for j = 1:p

        derive = (X*theta_prev - y)'*X(:,j)/m;
        theta(j) -= alpha*derive;
    end
```

* Vectorized version

```m
theta -= alpha*X'*(X*theta-y)/m;
```

# 3. Linear regression with multiple variable

## Feature normalization
When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly.

* Non-vectorized version

```m
function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

for p = 1:size(X, 2)
    mu(p) = mean(X(:, p), "a");
    sigma(p) = std(X(:, p));
end

for p = 1:size(X, 2)
    for i = 1:size(X, 1)
      X_norm(i, p) = (X(i, p)-mu(p))/sigma(p);
    end
end

% ============================================================

end

````

* Vectorized version

```m
    mu = mean(X, "a");
    sigma = std(X);

    ones_matrix = ones(size(X));
    X_norm = (X - ones_matrix*diag(mu))./(ones_matrix*diag(sigma));
```

Estimate the price of a 1650 sq-ft, 3 br house, in ex1_multi.m

```m
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = [1 ([1650 3] - mu)./sigma]*theta; % You should change this
```

## Normal Equations

The closed-form solution to linear regression is
\\[\theta = (X^TX)^{-1}X^T\vec{y}\\]

* Implementation

```m
function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------


theta = inv(X'*X)*X'*y;

% -------------------------------------------------------------


% ============================================================

end
```

Estimate the price of a 1650 sq-ft, 3 br house

```m
price = [1 1650 3]*theta; % You should change this
```