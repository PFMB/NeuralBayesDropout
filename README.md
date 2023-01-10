# Bayesian Neural Network for Uncertainty Quantification

This project designs a Bayesian Neural Network (BNN) with MCDropout and applies it to the MNIST data set. The objective of this task is to get a reasonable calibration and accuracy with the designed BNN. The challenge is that the train and the test data do not come from the same distribution. The test images exhibit varying degrees of ambiguity and are further rotated by random angles. The ambiguity and the rotations introduce aleatoric and epistemic uncertainty since the utilized network only observes non-rotated images during training.

In a nutshell, a Bayesian Neural Network specifies a prior distribution over the weights and uses neural networks to parametrize the likelihood functions. The simplest example would be a Gaussian prior on the weights

$$p(\theta) = N(0, \sigma^2_p I)$$

together with a Gaussian likelihood for the labels.

$$p(y| \textbf{x}, \theta) = N (f(\textbf{x}, \theta), \sigma^2)$$

Here, $f(.,.)$ is a neural network which is non-linear in the parameters. The vector $\theta$ collects the weights of the network and can be very high-dimensional. In general, the resulting posteriors over the weights are composed of high-dimensional integrals such that the predictions of $y$ become intractable which in turn requires approximate inference approaches (e.g. Variational Inference or MC Dropout). Further, a more complex likelihood also considered later is to model heteroscedastic noise with neural networks

$$p(y|\textbf{x},\theta) = N(f_1(\textbf{x},\theta), exp(f_2(\textbf{x},\theta)))$$

where $f_1(.,.)$ and $f_2(.,.)$ model the mean and log variance as two different outputs of a neural network.

### MC Dropout

A specialized approximate inferene approach tailored for BNNs is Monte Carlo (MC) Dropout. Dropout is a regularization technique in neural networks that randomly eliminates ("drop out") hidden units during each epoch of the optimization routine with probability $p$. As a result, the correpsonding weights $\theta_j$ are either set to 0 with probability p or $\lambda_j$ with probability $1-p$. Dropout can then be viewed as performing variational inference with a partiuclar variational family

$$q(\theta|\lambda) = \Pi_j q_j(\theta_j|\lambda_j)$$

where $q_j(\theta_j|\lambda_j) = p \delta_0(p_j) + (1-p)\delta_{\lambda_j}(\theta_j)$ and $\delta$ is the dirac measure wrt the point mass. The predictive uncertainty can then be approximated via

$$p(y^{\ast}|\textbf{x}^{\ast}, \textbf{x}_{1:n}, y_{1:n}) \approx \frac{1}{m} \sum_{j=1}^{m} p(y^{\ast}|\textbf{x}^{\ast}, \theta_j)$$

where $y^{\ast}$ are the labels on the test set and $\textbf{x}^*$ the corresponding predictors. The weight vector $\theta_j$ then has dimensions set to 0 with probability $p$. As a result, MC Dropout give a natural and computational efficient way on how to quantify uncertainty in a neural network. For the computational implementation, at prediction, ones needs to sample from the trained model multiple times for Monte Carlo integration. Bear in mind that in regular neural nets, at test, you use all neurons, but premultiply the output of the last hidden layer with $1-p$ to get the same expected value as during training.

### Expected Calibration Error

A intuitive measure for the degree and quality of calibration is the Expected Calibration Error (ECE). It measures the absolute difference between the true accuracy $P(\hat{Y} = Y| \hat{P} = p)$ and the confidence $p$, that is, 

$$E_{p\sim\hat{P}}(|P(\hat{Y} = Y| \hat{P} = p) - p|)$$

The true accuracy is approximated via a binning over $M$ intervals leading to the empirical accuracy. $B_m$ denotes the set of indices whose predicted confidence

$$\hat{p}_{i}$$ 

falls into the Interval $I_{m} = (\frac{m-1}{M}, \frac{m}{M}]$. The empirical confidence is defined as 

$$conf(B_m) = \frac{1}{|B_m|} \sum_{i\in B_m} \hat{p}_i$$

and the empirical accuracy is defined as

$$acc(B_{m}) = \frac{1}{|B_{m}|} \sum_{i\in B_{m}} 1_{\hat{y}_{i} = y_{i}}$$

resulting in the empirical Expected Calibration Error

$$\hat{ECE} = \sum_{m = 1}^{M} |acc(B_m) - conf(B_m)|$$

TODO: Show reliability diagrams

### Results

Examples on the prediction on the ambiguous and rotated MNIST test data:

![](https://github.com/PFMB/NeuralBayesDropout/blob/main/ambiguous_rotated_mnist.jpg)

Examples where the BNN is most and least confident wrt to the classification on Fashion MNIST:

![](https://github.com/PFMB/NeuralBayesDropout/blob/main/fashionmnist_least_confident.jpg)

![](https://github.com/PFMB/NeuralBayesDropout/blob/main/fashionmnist_most_confident.jpg)

This project was part of the Probabilistic Artificial Intelligence course at ETH. The skeleton was provided through the course while the core functionality was implemented by the author.