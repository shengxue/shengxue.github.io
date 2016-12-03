---
layout: post
title: Exercise 4 - Neural Network Learning
description: Exercise 4 Neural Networks of the stanford Corsera course Machine learning
modified: 2016-11-07
tags: [Machine learning, Stanford, Assignment]
---

# 1. Neural Networks

## Feedforward and const function

The cost function for the neural network (without regularization) is

\\[
J(\theta) = \frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K
\bigg[
−y_k^{(i)}\log((h_{θ}(x^{(i)}))_k)−(1−y_k^{(i)})\log(1−(h_θ(x^{(i)}))_k)
\bigg]
\\],

where \$h_{\theta}(x^{(i)})\$ is computed as shown in the Figure 2 and \$K = 10\$ is the total number of possible labels. Note that \$h_θ(x^{(i)})_k = a^{(3)}_k\$ is the activation (output value) of the \$k-th\$ output unit.

**Implementation-nnCostFunction.m**

```m
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);

yd = eye(num_labels);
y = yd(y,:);

log_dif = -log(a3).*y-log(1-a3).*(1-y);
J=sum(log_dif(:))/m;

```

## Regularized const function

The cost function for neural networks with regularization is given by

\\[
J(\theta) = \frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K
\bigg[
−y_k^{(i)}\log((h_{θ}(x^{(i)}))_k)−(1−y_k^{(i)})\log(1−(h_θ(x^{(i)}))_k)
\bigg]
\\]

\\[
+ \frac{\lambda}{2m}\bigg[\sum_{j=1}^{25}\sum_{k=1}^{400}(\theta_{j,k}^{(1)})^2+\sum_{j=1}^{10}\sum_{k=1}^{25}(\theta_{j,k}^{(2)})^2\bigg]
\\]

**Implementation-nnCostFunction.m**

```m
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);

yd = eye(num_labels);
y = yd(y,:);

log_dif = -log(a3).*y-log(1-a3).*(1-y);

Theta1s=Theta1(:,2:end);
Theta2s=Theta2(:,2:end);

penalty = lambda/(2*m)*(sum((Theta1s.*Theta1s)(:)) + sum((Theta2s.*Theta2s)(:)));
J=sum(log_dif(:))/m + penalty;
```

# 2. Backpropagation

## 