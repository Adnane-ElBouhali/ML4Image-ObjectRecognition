# Introduction Supervised Learning - Theoretical questions

---

## OLS (Ordinary Least Squares)

We have seen that the OLS estimator is equal to $\beta^* = (X^TX)^{-1}X^Ty$ which can be rewritten as $\beta^* = Hy$. Let $\hat{\beta} = Cy$ be another linear unbiased estimator of $\beta$ where $C$ is a $d \times n$ matrix, e.g., $C = H + D$ where $D$ is a non-zero matrix.

- Demonstrate that OLS is the estimator with the smallest variance: compute $E[\hat{\beta}]$ and $Var(\hat{\beta}) = E[(\hat{\beta} - E[\hat{\beta}])(\hat{\beta} - E[\hat{\beta}])^T]$ and show when and why $Var(\beta^*) < Var(\hat{\beta})$. Which assumption of OLS do we need to use?

### Answer

To demonstrate why the Ordinary Least Squares (OLS) estimator has the smallest variance among linear unbiased estimators, we will leverage the Gauss-Markov theorem. This theorem posits that under certain assumptions, the OLS estimator is the Best Linear Unbiased Estimator (BLUE), meaning it has the smallest variance among all unbiased linear estimators of the coefficients in linear regression. These assumptions include:

1. **Linearity of parameters**: The model is linear in parameters.
2. **Random sampling**: The data consists of a random sample from the population.
3. **No perfect multicollinearity**: The explanatory variables are not perfectly linearly related.
4. **Zero conditional mean**: The expectation of the error terms, given any value of the explanatory variables, is zero.
5. **Homoscedasticity**: The error terms have constant variance.

Given two estimators, $\beta^* = Hy$ for OLS and an alternative $\hat{\beta} = Cy$, where $C = H + D$ and $D$ is a non-zero matrix, let's explore their expectations and variances under these assumptions.

#### Expectation
For $\beta^*$, the expected value is derived as follows:
$$ E[\beta^*] = E[Hy] = H E[y] = HX\beta $$
Given $H = (X^TX)^{-1}X^T$, and thus $HX = I$ (the identity matrix), we have $E[\beta^*] = \beta$, confirming that $\beta^*$ is an unbiased estimator.

For $\hat{\beta}$, the expected value is:
$$ E[\hat{\beta}] = E[Cy] = CE[y] = CX\beta $$
To maintain unbiasedness, it's required that $CX = I$. This indicates that for $\hat{\beta}$ to be unbiased, $C$ must be chosen such that it, too, results in an unbiased estimator, akin to $H$.

#### Variance
The variance of $\hat{\beta}$ is calculated as follows:
$$ Var(\hat{\beta}) = E[(\hat{\beta} - E[\hat{\beta}])(\hat{\beta} - E[\hat{\beta}])^T] $$
Simplifying this expression, we obtain:
$$ Var(\hat{\beta}) = CE[\epsilon\epsilon^T]C^T = \sigma^2CC^T $$
where $\epsilon$ represents the error terms. Assuming homoscedasticity and independence of errors, $E[\epsilon\epsilon^T] = \sigma^2I$, leading to the variance formula above.

Comparing $Var(\beta^*)$ and $Var(\hat{\beta})$:
- $Var(\beta^*) = \sigma^2HH^T = \sigma^2(X^TX)^{-1}$
- $Var(\hat{\beta}) = \sigma^2CC^T = \sigma^2(H + D)(H + D)^T = \sigma^2(HH^T + HD^T + DH^T + DD^T)$

Since $D$ is non-zero and $HD^T + DH^T + DD^T$ adds positive values to $HH^T$, the variance of $\hat{\beta}$ will generally be larger than that of $\beta^*$. Thus, under the Gauss-Markov theorem's assumptions, $\beta^*$, the OLS estimator, indeed has the smallest variance among all linear unbiased estimators. This variance minimization is a critical feature that justifies the preference for the OLS method in linear regression analysis under the classical linear regression model.

---

## Ridge Regression

Suppose that both $y$ and the columns of $x$ are centered ($y_c$ and $x_c$) so that we do not need the intercept $\beta_0$. In this case, the matrix $x_c$ has $d$ (rather than $d+1$) columns. We can thus write the criterion for ridge regression as:

$$\beta^*_{\text{ridge}} = \arg\min_{\beta} \left\{ (y_c - x_c\beta)^T(y_c - x_c\beta) + \lambda\|\beta\|^2 \right\}$$

- Show that the estimator of ridge regression is biased (that is $E[\beta^*_{\text{ridge}}] \neq \beta$).

### Answer:

In ridge regression, the estimator $\beta^*_{\text{ridge}}$ is derived by minimizing the penalized residual sum of squares, where the penalty is proportional to the square of the magnitude of the coefficients. This approach is intended to reduce overfitting by shrinking the coefficients towards zero. The penalized residual sum of squares can be expressed as:

$$ \beta^*_{\text{ridge}} = \arg\min_{\beta} \left\{ (y_c - x_c\beta)^T(y_c - x_c\beta) + \lambda\|\beta\|^2 \right\} $$

The solution to this optimization problem is given by:

$$ \beta^*_{\text{ridge}} = (x_c^Tx_c + \lambda I)^{-1}x_c^Ty_c $$

To understand the bias in $\beta^*_{\text{ridge}}$, we look at its expectation:

$$ E[\beta^*_{\text{ridge}}] = E\left[(x_c^Tx_c + \lambda I)^{-1}x_c^Ty_c\right] $$

Given that $y_c = x_c\beta + \epsilon$, with $\epsilon$ representing the error term, and assuming $E[\epsilon] = 0$, we substitute $y_c$ in the expression for the expectation:

$$ E[\beta^*_{\text{ridge}}] = E\left[(x_c^Tx_c + \lambda I)^{-1}x_c^T(x_c\beta + \epsilon)\right] $$

Distributing $x_c^T$ inside the expectation yields:

$$ E[\beta^*_{\text{ridge}}] = (x_c^Tx_c + \lambda I)^{-1}x_c^Tx_c\beta + (x_c^Tx_c + \lambda I)^{-1}x_c^TE[\epsilon] $$

Since we assume $E[\epsilon] = 0$, this simplifies to:

$$ E[\beta^*_{\text{ridge}}] = (x_c^Tx_c + \lambda I)^{-1}x_c^Tx_c\beta $$

It's clear that $E[\beta^*_{\text{ridge}}]$ does not equal $\beta$ directly due to the presence of $\lambda I$ in the inverse term, which introduces bias. The term $(x_c^Tx_c + \lambda I)^{-1}x_c^Tx_c$ does not equal the identity matrix unless $\lambda = 0$. This deviation indicates that the ridge regression estimator $\beta^*_{\text{ridge}}$ is biased. The bias is a trade-off for reduced variance and better generalization to new data, particularly in scenarios with multicollinearity or when the number of predictors exceeds the number of observations.


- Recall that the SVD decomposition is $x_c = UDV^T$. Write down by hand the solution $\beta^*_{\text{ridge}}$ using the SVD decomposition. When is it useful using this decomposition? Hint: do you need to invert a matrix?

### Answer:

When incorporating the Singular Value Decomposition (SVD) of $x_c$ into the formula for $\beta^*_{\text{ridge}}$, we utilize the fact that SVD breaks down the matrix $x_c$ into three components: $U$, a matrix of orthogonal eigenvectors of $x_cx_c^T$; $D$, a diagonal matrix with the singular values of $x_c$; and $V$, a matrix of orthogonal eigenvectors of $x_c^Tx_c$. This decomposition is powerful for simplifying the expression for $\beta^*_{\text{ridge}}$ and making the computation more efficient and numerically stable. The initial step is to substitute the SVD components into the ridge regression formula:

$$ \beta^*_{\text{ridge}} = (V D U^T U D V^T + \lambda I)^{-1} V D U^T y_c $$

Given the properties of orthogonal matrices, where $U^TU = I$ and $VV^T = I$, with $I$ being the identity matrix, we can simplify the above expression:

$$ \beta^*_{\text{ridge}} = (V D^2 V^T + \lambda I)^{-1} V D U^T y_c $$

The ridge regression estimator further simplifies due to the diagonal structure of $D^2$ and the orthogonality of $U$ and $V$:

$$ \beta^*_{\text{ridge}} = V(D^2 + \lambda I)^{-1}DV^T y_c $$

This formulation leverages the diagonal nature of $D^2 + \lambda I$ for computational benefits:

1. **Numerical Stability**: Direct inversion of $x_c^Tx_c$ in the presence of multicollinearity or high dimensionality (large $d$) can lead to numerical instability due to the near-singular or ill-conditioned nature of $x_c^Tx_c$. Utilizing SVD avoids these issues since the inverse of a diagonal matrix (augmented by $\lambda I$) is inherently well-conditioned, ensuring more stable computations.

2. **Computational Efficiency**: Inverting a large matrix directly is computationally intensive. However, the SVD transformation simplifies the inversion process to just dealing with the diagonal elements of $D^2 + \lambda I$, significantly reducing computational complexity. This aspect is particularly advantageous for large-scale data or when dealing with matrices that are difficult to invert directly.

Thus, the SVD approach in ridge regression not only improves numerical stability but also enhances computational efficiency by simplifying the inversion process to straightforward operations on diagonal matrices, making it an effective method for computing the ridge regression estimator in practical applications.


- Remember that $Var(\beta^*_{\text{OLS}}) = \sigma^2(X^TX)^{-1}$. Show that $Var(\beta^*_{\text{OLS}}) \geq Var(\beta^*_{\text{ridge}})$.

### Answer:

The OLS estimator's variance is given by:

$$ Var(\beta^*_{\text{OLS}}) = \sigma^2(X^TX)^{-1} $$

For the ridge regression estimator, leveraging the Singular Value Decomposition (SVD) of $X$ allows for an efficient expression. Represented as $X = UDV^T$, the ridge regression solution becomes:

$$ \beta^*_{\text{ridge}} = V(D^2 + \lambda I)^{-1}DV^T y $$

This results in the variance of the ridge regression estimator as:

$$ Var(\beta^*_{\text{ridge}}) = \sigma^2V(D^2 + \lambda I)^{-2}V^T $$

The goal is to demonstrate the relationship:

$$ \sigma^2(X^TX)^{-1} \geq \sigma^2V(D^2 + \lambda I)^{-2}V^T $$

Given the SVD, $X^TX$ translates to $VD^2V^T$, and thus, the OLS variance simplifies to:

$$ Var(\beta^*_{\text{OLS}}) = \sigma^2(VD^2V^T)^{-1} $$

By isolating the inverse on one side, we illustrate:

$$ VD^2V^T \cdot Var(\beta^*_{\text{OLS}}) = \sigma^2I $$

The positive semi-definite nature of $VD^2V^T$ ensures that $Var(\beta^*_{\text{OLS}})$ also adheres to this characteristic, leading to:

$$ VD^2V^T \cdot Var(\beta^*_{\text{OLS}}) \geq \sigma^2I $$

For the ridge regression variance:

$$ V(D^2 + \lambda I)^{-2} \cdot Var(\beta^*_{\text{ridge}}) = \sigma^2I $$

When we adjust for the regularization term by multiplying both sides with $(D^2 + \lambda I)^{2}$:

$$ Var(\beta^*_{\text{ridge}}) = \sigma^2V(D^2 + \lambda I)^{-2}V^T $$

Recognizing that each element $d_i^2 + \lambda$ in $(D^2 + \lambda I)$ exceeds its counterpart in $D^2$, their inverses reflect a smaller magnitude, leading to:

$$ V(D^2 + \lambda I)^{-2}V^T \leq VD^{-2}V^T $$

Hence, when scaling by $\sigma^2$:

$$ \sigma^2V(D^2 + \lambda I)^{-2}V^T \leq \sigma^2V(D^2V^T)^{-1} $$

Illustrating that:

$$ Var(\beta^*_{\text{ridge}}) \leq Var(\beta^*_{\text{OLS}}) $$

This analysis showcases that the variance of the OLS estimator is at least as large as, if not larger than, the variance of the ridge regression estimator, reflecting the trade-off between bias and variance introduced by the regularization parameter $\lambda$ in ridge regression.

- When $\lambda$ increases what happens to the bias and to the variance? Hint: Compute MSE = $E[(y_0 - x_0^T\beta^*_{\text{ridge}})^2]$ at the test point $(x_0, y_0)$ with $y_0 = x_0^T\beta + \epsilon_0$ being the true model and $\beta^*_{\text{ridge}}$ the ridge estimate.

### Answer :

Exploring the dynamics of bias and variance with increasing values of $\lambda$ in ridge regression involves a detailed look at the Mean Squared Error (MSE) at a particular test point $(x_0, y_0)$. The MSE formula encapsulates the essence of the bias-variance trade-off and is given by:

$$
MSE = Var(x_0^T\beta^*_{\text{ridge}}) + [Bias(x_0^T\beta^*_{\text{ridge}})]^2 + Var(\epsilon_0)
$$

With $y_0$ defined as $x_0^T\beta + \epsilon_0$, where $x_0$ represents a new observation not included in the training set and $\epsilon_0$ signifies the associated error term, the bias of the ridge regression estimator at this specific point is captured by:

$$
Bias(x_0^T\beta^*_{\text{ridge}}) = E[x_0^T\beta^*_{\text{ridge}}] - x_0^T\beta
$$

A key observation here is the behavior of the ridge estimator $\beta^*_{\text{ridge}}$ as $\lambda$ is increased. Specifically:

- **Increasing $\lambda$** directly influences the ridge estimator by compelling it towards zero. This phenomenon has a dual effect on both the bias and variance components of the MSE:
    - The **Bias** increases due to the fact that as $\beta^*_{\text{ridge}}$ approaches zero, the expectation $E[x_0^T\beta^*_{\text{ridge}}]$ drifts away from the actual $x_0^T\beta$, thereby magnifying the difference between the estimated and true model outputs at $x_0$.
    - The **Variance** sees a reduction as a consequence of the regularization term $\lambda$ acting on the coefficients. The mechanism behind this is related to the impact of $\lambda$ on the matrix $(D^2 + \lambda I)$, where an increase in $\lambda$ results in the diagonal elements of this matrix growing larger, which in turn leads to a decrease in the elements of its inverse, $(D^2 + \lambda I)^{-2}$, thereby reducing the overall variance.

This nuanced interplay between bias and variance as $\lambda$ increases underlines the core principle of the bias-variance trade-off in ridge regression. The challenge lies in selecting an optimal $\lambda$ that harmonizes these two opposing forces to minimize the overall MSE, thereby enhancing the model's predictive accuracy on unseen data.


- Show that $\beta^*_{\text{ridge}} = \frac{\beta^*_{\text{OLS}}}{1+\lambda}$ when $X^TX = I_d$

### Answer :

When we compare the Ordinary Least Squares (OLS) estimator with the ridge regression estimator under a specific mathematical condition, we gain insights into how regularization affects parameter estimation in linear regression. The OLS estimator is traditionally defined as:

$$ \beta^*_{\text{OLS}} = (X^TX)^{-1}X^Ty $$

In contrast, the ridge regression estimator introduces a regularization term, $\lambda$, which penalizes large coefficients:

$$ \beta^*_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^Ty $$

Given a special condition where $X^TX = I_d$, the identity matrix, we can simplify these expressions further. For the OLS estimator, under this condition, the formula simplifies to $X^Ty$, since multiplying by the identity matrix does not change the value:

$$ \beta^*_{\text{OLS}} = I_d^{-1}X^Ty = X^Ty $$

For the ridge regression estimator, the presence of $\lambda I_d$ alters the estimation process by adding a degree of bias, with the intention of reducing variance and potentially improving prediction accuracy on new data:

$$ \beta^*_{\text{ridge}} = (I_d + \lambda I_d)^{-1}X^Ty $$

Given that $I_d + \lambda I_d$ is a diagonal matrix where each diagonal element is $1 + \lambda$, the inverse of this matrix is another diagonal matrix where each element is $\frac{1}{1+\lambda}$. Thus, applying this inverse to $X^Ty$, we get:

$$ \beta^*_{\text{ridge}} = \frac{1}{1+\lambda}I_dX^Ty $$

Recognizing that $I_dX^Ty$ simplifies to $X^Ty$, the relationship between the ridge regression estimator and the OLS estimator under this condition is:

$$ \beta^*_{\text{ridge}} = \frac{1}{1+\lambda}\beta^*_{\text{OLS}} $$

This elucidates a key aspect of ridge regression: as $\lambda$ increases, the estimated coefficients are shrunk towards zero relative to the OLS estimates. This shrinkage is a direct result of the regularization process, which aims to reduce the complexity of the model by penalizing the magnitude of the coefficients, thereby potentially enhancing the model's generalizability to new, unseen data.

---

## Elastic Net

Using the previous notation, we can also combine Ridge and Lasso in the so-called Elastic Net regularization:

$$ \beta^*_{\text{ENet}} = \arg\min_{\beta} \{ (y_c - x_c\beta)^T(y_c - x_c\beta) + \lambda_2\|\beta\|^2 + \lambda_1\|\beta\|_1 \} $$

Calling $\alpha = \frac{\lambda_2}{\lambda_1+\lambda_2}$, solving the previous Eq. is equivalent to:

$$ \beta^*_{\text{ENet}} = \arg\min_{\beta} \{ (y_c - x_c\beta)^T(y_c - x_c\beta) + \lambda (\alpha \sum_{j=1}^{d} \beta_j^2 + (1 - \alpha) \sum_{j=1}^{d} |\beta_j|) \} $$

- This regularization overcomes some of the limitations of the Lasso, notably:
  - If $d > N$ Lasso can select at most $N$ variables → ENet removes this limitation.
  - If a group of variables are highly correlated, Lasso randomly selects only one variable → with ENet correlated variables have a similar value (grouped).
  - Lasso solution paths tend to vary quite drastically → ENet regularizes the paths.
  - If $N > d$ and there is high correlation between the variables, Ridge tends to have a better performance in prediction → ENet combines Ridge and Lasso to have better (or similar) prediction accuracy with less (or more grouped) variables.


- Compute by hand the solution of Eq.2 supposing that $X_c^TX_c = I_d$ and show that the solution is:

$$ \beta^*_{\text{ENet}} =  \frac{(\beta^*_{\text{OLS}})_j \pm \frac{\lambda_1}{2}}{1+\lambda_2} $$

### Answer :

Arriving at the Elastic Net solution, which integrates both L1 and L2 regularization approaches, involves handling the optimization problem under the assumption that the predictors are orthogonal (i.e., $X_c^T X_c = I_d$, the identity matrix). The Elastic Net objective function, given this assumption, is expressed as:

$$
\beta^{ENet} = \arg \min_{\beta} \{ ||y_c - X_c\beta||^2_2 + \lambda_2||\beta||^2_2 + \lambda_1||\beta||_1 \}
$$

Given the orthogonality condition, this simplifies the analysis significantly. For the Ridge regression component (with $\lambda_1 = 0$), the solution is a scaled version of the Ordinary Least Squares (OLS) estimator:

$$
\beta^{Ridge} = \frac{\beta^{OLS}}{1+\lambda_2}
$$

In the context of Lasso regression, which emphasizes an L1 penalty, a soft-thresholding operation is applied to each coefficient. For a specific coefficient $j$, the operation, considering orthogonality, is defined as:

$$
S_{\lambda_1}((\beta^{OLS})_j) = \text{sign}((\beta^{OLS})_j)\left(|(\beta^{OLS})_j| - \frac{\lambda_1}{2}\right)_+
$$

where $(x)_+$ denotes the positive part of $x$, and $\text{sign}(x)$ indicates the sign of $x$.

The Elastic Net approach combines the effects of both penalties. For each coefficient, it integrates the Lasso's soft-thresholding followed by Ridge's shrinkage factor:

$$
\beta^{ENet}_j = \frac{S_{\lambda_1}((\beta^{OLS})_j)}{1+\lambda_2}
$$

Inserting the soft-thresholding formula, we obtain:

$$
\beta^{ENet}_j = \frac{\text{sign}((\beta^{OLS})_j)\left(|(\beta^{OLS})_j| - \frac{\lambda_1}{2}\right)_+}{1+\lambda_2}
$$

The effect on $\beta^{ENet}_j$ depends on the magnitude and direction of $(\beta^{OLS})_j$:

- If $(\beta^{OLS})_j > \frac{\lambda_1}{2}$, the sign function $\text{sign}((\beta^{OLS})_j)$ equals $+1$.
- If $(\beta^{OLS})_j < -\frac{\lambda_1}{2}$, the sign function equals $-1$.
- For $|(\beta^{OLS})_j| \leq \frac{\lambda_1}{2}$, the outcome of soft-thresholding is zero.

Hence, for non-zero $\beta^{ENet}_j$ coefficients, the formulation can be interpreted as:

$$
\beta^{ENet}_j = \frac{(\beta^{OLS}_j) \pm \frac{\lambda_1}{2}}{1+\lambda_2}
$$

The choice of $\pm$ is dictated by the original OLS coefficient's sign, demonstrating how the Lasso effectively adjusts the coefficient by either subtracting or adding $\frac{\lambda_1}{2}$, followed by applying the Ridge's proportionate shrinkage through $\frac{1}{1+\lambda_2}$. This illustrates the Elastic Net's nuanced balance between encouraging sparsity via the L1 penalty and ensuring stability through the L2 penalty.
