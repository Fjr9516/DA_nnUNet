from .functional import revgrad
from torch.nn import Module
from torch import tensor


class RevGrad(Module):
    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)

    def get_alpha(self):
        return self._alpha.numpy()

    def set_alpha(self, new_alpha: float):
        self._alpha = tensor(new_alpha, requires_grad=False)

