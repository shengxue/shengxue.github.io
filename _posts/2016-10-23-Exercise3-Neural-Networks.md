---
layout: post
title: Exercise 3 - Neural Networks
description: Exercise 3 Neural Networks of the standord Corsera course Machine learning
modified: 2016-10-23
tags: [Machine learning, Stanford, Assignment]
---

Logistic regression cannot form more complex hypotheses as it is only a linear classifier[^1].

In this exercise, the set of network parameters (\$\theta^{(1)}, \theta^{(2)}\$) are already trained and provided in **ex3weights.mat**

[predict.m](https://github.com/shengxue/machine-learning-assignment/blob/master/machine-learning-ex3/ex3/predict.m)

```m
function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);

[p_max, p] = max(a3, [], 2);

% =========================================================================

end

```

[^1]:We could add more features (such as polynomial features) to logistic regression, but that can be very expensive to train.