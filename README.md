# bayesian-sgd-research-for-panel-economic-data
This project is designed to research the effectiveness of Bayesian Optimized Stochastic Gradient Descent (SGD) Neural Networks on panel economic analysis, using an SGD neural network (NN) model along with a Bayesian Optimization algorithm for model hyperparameter tuning.

The Bayesian Optimized SGD NN model was benchmark tested against the default MLP Regressor model from the sklearn.neural_network Python library, on panel economic data compiled from the US Census and Federal Reserve, designed to track real personal disposable income in the US using six features of economic activity indicators tracked by the US Census.

The results from the testing showed that the Bayesian Optimized SGD NN model achieved around an 84% lower RMSE on testing data compared to the MLP Regressor model, providing extremely significant statistical evidence that Bayesian Optimized SGD NN models can outperform standard NN models (t = -5.519, p = 1.194 * 10^-6 - approximately).

Upon utilizing a backward weighting technique on the trained parameters of the Bayesian Optimized SGD model across trials, it was shown that Construction and Housing, Consumer Spending, and Manufacturing, have the strongest positive effect on real personal disposable income in the US.
