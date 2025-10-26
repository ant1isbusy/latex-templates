
#import "@preview/frame-it:1.2.0": *

#let (example, feature, variant, syntax) = frames(
  feature: ("Feature",),
  variant: ("Variant",),
  example: ("Example", rgb("#1eb0f9")),
  syntax: ("Syntax",),
)

#show: frame-style(styles.boxy)

#set page(
  paper: "a5",
  margin: (x: 1.8cm, y: 1.5cm),
)

#set par(
  justify: true,
  leading: 0.52em,
)

#set text(
  size: 11pt,
)

#let gauss = $frac(1, root(, 2 pi sigma^2)) exp(-frac((x-mu)^2, 2sigma^2))$

= Deep Learning Notes

=== Statistical Learning Theory Recap
We assume all data stems from the data generating distribution $p_"data"$. Our goal is to achieve a minimal generalization error, so the error on randomly drawn samples from $p_"data"$.

Generalization error = $integral_x p(x) E(x|theta) d x$, where $p(x)$ is the probability of $x$ occuring and $E(x|theta)$ is the expected loss for input x given model parameters $theta$.

=== Estimators

Again, there’s some (unknown) probability distribution in the real world that generates the inputs $x$ and targets $t$.

Our goal is to approximate this true distribution with a *model*.

Notationwise (regarding "|"):

- $p(t|x)$: *conditional* distribiton, tells us how likely target t is given input x.
- $p(x,t)$: *joint* distribution, tells us how likely input x and target t are to occur together, where $p(x,t)=p(t|x) dot p(x)$.

We can therefore write down models using this probabilistic notation:

A discriminative model: $p_"model" (t|x; theta) = p_theta (t|x; theta)$: For every given $theta$ you get a specific way of assigning probabilities to targets t given inputs x.

A generative model: $p_"model" (x,t|theta)$: Given $theta$ how do we generate inputs x and targets t together. That is $p_"model" ((x,t)|theta)$.

Training is the estimation of the parameter *$theta$*.

=== Point Estimators

A point estimator gives a single best guess for some unknown, like a parameter in our model. It is calculated from a sample of the data: $accent(theta, hat)_m = g(bold(x)^((1)),...,bold(x^((m))))$, so a point estimator is the function $g$.

=== Bias and Variance

The bias of an estimator $"bias"(accent(theta, hat)_m)=bb(E)[accent(theta, hat)_m]-theta$.

The variance of an estimator $"var"(accent(theta, hat)_m)=bb(E)[(accent(theta, hat)_m - bb(E)[accent(theta, hat)_m])^2]$.

An estimator is consistent if it converges to the true value as the number of samples $m$ goes to infinity.

=== Maximum Likelihood Estimation (MLE)

We observe some i.i.d data drawn from $p_"data"$. Let $p_"model"(bold(x), theta)$ be a parametric family of distributions. The maximum likelihood estimate is then:

$
  bold(theta)_"ML" & = arg max_theta p_"model" (bold(X); theta) \
                   & = arg max_theta product_(i=1)^m p_"model" (bold(x)^((i)); theta) \
                   & = arg max_theta sum_(i=1)^m log p_"model" (bold(x)^((i)); theta)
$


#example[MLE for Gaussian][
  Let our model try to learn the mean $mu$ of a Gaussian with known variance $sigma^2$. The model is therefore:
  $ p_"model" (x|mu, sigma) = gauss $
  $
    bold(accent(mu, hat))_"ML" &= arg max_mu product_(i=1)^m p_"model" (bold(x)^((i))|mu, sigma) \
    &= arg max_mu sum_(i=1)^m log p_"model" (bold(x)^((i))|mu, sigma) \
    &= arg max_mu sum_(i=1)^m (log frac(1, root(, 2 pi sigma^2)) - frac((bold(x)^((i))-mu)^2, 2sigma^2)) \
  $
  We can drop all terms that don’t depend on $mu$:
  $
    = arg min_mu sum_(i=1)^m (bold(x)^((i))-mu)^2
  $

  To find the minimum, we derive and set to zero:
  $
    frac(d, d mu) sum_(i=1)^m (bold(x)^((i))-mu)^2 = sum_(i=1)^m 2(mu - bold(x)^((i))) = 0 \
    = m mu + sum_(i=1)^m - bold(x)^((i)) = 0 \
    arrow mu = frac(1, m) sum_(i=1)^m bold(x)^((i)) \
  $
]

=== Maximum A Posteriori Estimation (MAP)

In Maximum Likelihood Estimation (MLE), we only look for parameters that make the observed data most likely. However, MLE ignores any prior knowledge we might have about the parameters.

MAP estimation extends this by combining the *likelihood* with a *prior belief* over parameters, using Bayes’ rule:
$
  p(theta|bold(X)) = frac(p(bold(X)|theta) p(theta), p(bold(X)))
$

The MAP estimate is the parameter value that maximizes the *posterior* probability:
$
  bold(theta)_"MAP" = arg max_theta p(theta|bold(X))
$

Since $p(bold(X))$ does not depend on $theta$, we can drop it:
$
  bold(theta)_"MAP" & = arg max_theta p_"model" (bold(X)|theta) p(theta) \
                    & = arg max_theta [log p_"model" (bold(X)|theta) + log p(theta)] \
                    & = arg max_theta [sum_(i=1)^m log p_"model" (bold(x)^((i))|theta) + log p(theta)] \
$


This makes the difference between MLE and MAP the following:

- *MLE* says: pick parameters that best explain the data, even if they’re huge or weird.
- *MAP* says: pick parameters that explain the data and make sense according to what we believe is reasonable (small weights, smoother functions, etc.).

#example[Gaussian Prior][
  Suppose we believe that the parameters (e.g. neural network weights) should be *close to zero* rather than very large.
  We can express this belief using a Gaussian prior:
  $
    p(theta) = gauss[theta; 0, frac(1, lambda) bold(I)]
  $
  This prior assigns highest probability to small values of $theta$, and penalizes large weights.

  Taking the logarithm of the prior:
  $
    log p(theta) = c - frac(lambda, 2) ||theta||^2
  $

  Plugging this into the MAP objective:
  $
    bold(theta)_"MAP" = arg max_theta [log p(bold(X)|theta) - frac(lambda, 2) ||theta||^2]
  $
  which is equivalent to minimizing:
  $
    bold(theta)_"MAP" = arg min_theta [- log p(bold(X)|theta) + frac(lambda, 2) ||theta||^2]
  $

  Therefore, MAP estimation with a Gaussian prior is the same as adding *L2 regularization* (weight decay) to the loss function.

  The first term fits the data, and the second term keeps parameters small — balancing data fit and model simplicity.
]

=== Bias-Variance Tradeoff
The bias–variance tradeoff describes how an estimator’s total error (its mean squared error, MSE) is composed of two parts:
$
  "MSE" = ("Bias"[hat(theta)])^2 + "Var"[hat(theta)].
$
High bias means the model’s predictions are systematically off (underfitting), while high variance means they fluctuate strongly across different datasets (overfitting).
Improving one typically worsens the other, so learning algorithms aim to minimize MSE by balancing these two sources of error.

== Classification and Decision Theory
=== Classification

We assume $K$ possible classes $C_1, …, C_K$ and an input vector $x in bb(R)^N$ drawn from
$P(x, C) = P(x|C) P(C)$.
Here, $P(x|C)$ is the class-conditional probability and $P(C)$ the class prior.

Our goal is to infer the posterior probability:

$
  P(C_k|x) = frac(P(x|C_k) P(C_k), sum_j P(x|C_j) P(C_j))
$

The optimal decision rule that minimizes misclassification rate is:

$
  y(x) = arg max_k P(C_k|x)
$


=== Binary Classification

For two classes $C_1$ and $C_2$, define

$
  a = ln frac(P(x|C_1) P(C_1), P(x|C_2) P(C_2))
$

The posterior can be expressed via the sigmoid:

$
  sigma(a) = frac(1, 1 + exp(-a)), quad P(C_1|x) = sigma(a)
$


=== Multi-class Classification

For $K > 2$, we use the softmax function:

$
  P(C_k|x) = frac(exp(a_k), sum_(j=1)^K exp(a_j))
$

where $a_k$ are model logits.
Softmax ensures normalized, positive probabilities.


=== Cross-Entropy Loss

Given training data $(bold(x)^(m), t^(m))$ with $t^(m) in {0,1}$
and model output $y^(m) = P(C_1|bold(x)^(m))$,
maximum likelihood corresponds to minimizing:

$
  E = - sum_m [ t^(m) log(y^(m)) + (1 - t^(m)) log(1 - y^(m)) ]
$

This is the standard cross-entropy loss in classification networks.


=== Decision Theory

Decision theory formalizes optimal choices when errors have different costs.


=== Minimizing Misclassification Rate

If all errors are equally costly, maximize the probability of correct decisions:

$
  bb(E)["Correct"] = integral_(bb(R)^N) P(x) P(C_(y(x))|x) d x
$

The optimal (Bayes) classifier is:

$
  y(x) = arg max_k P(C_k|x)
$


=== Minimizing Expected Loss

When different errors have different costs, use a loss matrix $L_(k,j)$,
where $L_(k,j)$ is the loss for predicting class $C_j$ when the true class is $C_k$.

Expected loss:

$
  bb(E)["Loss"] =
  integral_(bb(R)^N) P(x)
  sum_k L_(k, y(x)) P(C_k|x) d x
$

Optimal decision rule:

$
  y(x) = arg min_j sum_k L_(k,j) P(C_k|x)
$

When $L_(k,j) = 1$ for $k != j$ and $0$ otherwise,
this reduces to the Bayes classifier.





