class Optimizer:
    def __init__(self, parameters, lr):
        """
        Base class for optimization algorithms.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate for the optimizer.
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.state = {}
    
    def step(self):
        """
        Perform a single optimization step.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def zero_grad(self):
        """
        Set gradients of all parameters to zero.
        """
        for p in self.parameters:
            p.grad *= 0.


class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        """
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate for the optimizer.
            momentum: Momentum hyperparameter.
            dampening: Dampening for momentum.
            weight_decay: L2 regularization weight decay factor.
            nesterov: Whether to apply Nesterov momentum (default False).
        """
        super().__init__(parameters, lr)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("For Nesterov momentum specify a positive momentum value with zero dampening.")
        self.nesterov = nesterov

    def step(self):
        """
        Perform a single optimization step using SGD.
        """
        if self.momentum:
            if "b_t" in self.state:
                for b_t, p in zip(self.state["b_t"], self.parameters):
                    g_t = p.grad
                    if self.weight_decay:
                        g_t = g_t + p.data * self.weight_decay
                    b_t *= self.momentum
                    b_t += (1 - self.dampening) * g_t
                    if self.nesterov:
                        g_t = g_t + self.momentum * b_t
                    else:
                        g_t = b_t
                    p.data -= self.lr * g_t
            else:
                self.state["b_t"] = []
                for p in self.parameters:
                    g_t = p.grad
                    if self.weight_decay:
                        g_t = g_t + p.data * self.weight_decay
                    self.state["b_t"].append(g_t.copy())
                    if self.nesterov:
                        g_t = g_t + self.momentum * g_t
                    p.data -= self.lr * g_t
        else:
            for p in self.parameters:
                if self.weight_decay:
                    p.data -= self.lr * (p.grad + p.data * self.weight_decay)
                else:
                    p.data -= self.lr * p.grad
