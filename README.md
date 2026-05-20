# Linear_Regression_Coding
Appling linear regression techniques using Python

## Summary
In this lab, I built a simple linear regression model using Python and PyTorch, following the ideas from Chapter 3.1–3.5 of the Dive into Deep Learning textbook. The goal was to understand the basics of supervised learning and how models are trained (Zhang et al., 2023).

First, I generated a synthetic dataset using the function synthetic_data(w, b, num_examples) (lines 10–18), with predefined parameters true_w = torch.tensor([2.0, -3.4]) and true_b = 4.2 (lines 20–23). The feature matrix X was created using a normal distribution, and the labels y were computed using a linear equation. To make the data more realistic, Gaussian noise was added. I also checked the first few samples using features[:5], labels[:5] (line 25) to verify that the data was generated correctly (Programmerall, n.d.; Zhang et al., 2023).

Next, I implemented mini-batch stochastic gradient descent using data_iter(batch_size, features, labels) (lines 28–40), where the data is shuffled to ensure randomness during training. The model parameters were initialized with small random weights and a zero bias (lines 45–48), which supports effective learning. I then defined the linear model linreg(X, w, b) (lines 51–53) to make predictions, and used the squared_loss(y_hat, y) function (lines 56–58) to measure how close the predictions were to the actual values (Nguyen & Widrow, 1990; Goodfellow et al., 2016).

Finally, I optimized the model using the sgd(params, lr, batch_size) function (lines 61–68). Gradients were computed using backpropagation, and the parameters were updated over several epochs (line 75). The training loss was printed during each epoch (line 84), showing steady improvement and convergence. After training, I compared the learned parameters to the true values (lines 88–94), confirming that the model successfully captured the underlying relationship. A scatter plot of the data (lines 97–105) also showed a clear linear trend (Ruder, 2016; Zhang et al., 2023).

Overall, this lab demonstrates important concepts such as data generation, mini-batch training, gradient-based optimization, and parameter estimation. It also highlights how linear regression serves as a foundation for more advanced machine learning methods and real-world applications (Zhang et al., 2023; Programmerall, n.d.; Goodfellow et al., 2016; Ruder, 2016; Nguyen & Widrow, 1990).

## References
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.

Nguyen, T., & Widrow, B. (1990). Improving the learning speed of 2-layer neural networks by choosing initial values of the adaptive weights. Proceedings of the International Joint Conference on Neural Networks, 3, 21–26. https://doi.org/10.1109/IJCNN.1990.137819

Programmerall. (n.d.). Pytorch study 2020 spring -1-linear regression. https://www.programmerall.com/article/56251665039/

Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747. https://arxiv.org/abs/1609.04747

Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into deep learning. Cambridge University Press. https://d2l.ai
