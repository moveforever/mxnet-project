
@register
class Ftrl(Optimizer):

    def __init__(self, lamda1=0.01, lamda2=1, alpha=0.1, beta=1, epsilon=1e-5, **kwargs):
        super(Ftrl, self).__init__(**kwargs)
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.alpha = alpha
        self.beta = beta


    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context),  # sigma
            zeros(weight.shape, weight.context),  # z
            zeros(weight.shape, weight.context))  # n

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        wd = self._get_wd(index)
        self._update_count(index)

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # accumulated g and delta initlization
        sigma, z, n = state


        #update sigma, z, n
        sigma[:] = (sqrt(n + grad * grad) - sqrt(n)) / self.alpha
        z[:] += grad - sigma * weight
        n[:] += grad * grad

        # update weight
        weight[:] = (sign(z) * self.lamda1 - z) / ((self.beta + sqrt(n))/self.alpha + self.lamda2) * (abs(z) > self.lamda1)
