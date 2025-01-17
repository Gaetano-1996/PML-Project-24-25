import torch
import torch.nn as nn
from scipy.stats import qmc
from utils import *

#### Basic DDPM ########################################################################################################


class DDPM_classic(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """

        super(DDPM_classic, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        mean = (
            1.0
            / torch.sqrt(self.alpha[t])
            * (
                xt
                - (self.beta[t])
                / torch.sqrt(1 - self.alpha_bar[t])
                * self.network(xt, t)
            )
        )
        std = torch.where(
            t > 0,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    def elbo(self, x0):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """

        # Sample time step t
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        return -nn.MSELoss(reduction="mean")(epsilon, self.network(xt, t))

    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo(x0).mean()


#### DIFFERENT SAMPLING STRATEGIES #####################################################################################

#### DDPM with low discrepancy sampling ################################################################################


# Low discrepancy sampling with Sobol sequence
def low_discrepancy_sobol(num_samples, t_max):
    # Generate Sobol low-discrepancy samples
    sampler = qmc.Sobol(d=1, scramble=True)
    # handling the base2
    M = math.log2(num_samples)
    m = math.ceil(M)
    # generating the actual sample
    samples = sampler.random_base2(m=m)
    # taking the exact number of samples
    if len(samples) > num_samples:
        samples = samples[:num_samples]
    # mapping them to integer
    return torch.tensor((samples * t_max).astype(int), dtype=torch.long)


# Low discrepancy sampling with simple method
def low_discrepancy_simple(num_sample, t_max):
    # number of samples required
    k = num_sample
    # generating u0 ~ Unif(0,1), random starting point of the sequence
    u0 = torch.rand(1)
    # defining the time steps for i in {1,...,k}
    t_i = (u0 + torch.arange(1, k + 1) / k) % 1
    # mapping them back to (0,t_max) and making them integers,
    # then add 1 so that final range is (1,t_max)
    t_indices = (t_i * t_max).to(torch.int64) + 1
    t = t_indices.unsqueeze(-1)
    return t


class DDPM_low_discrepancy(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2, sampler="simple"):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        sampler: str
            The low discrepancy sampler to use. Options are "simple" and "sobol"
        """

        super(DDPM_low_discrepancy, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

        # selecting the low discrepancy sampler
        self.sampler = sampler

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        mean = (
            1.0
            / torch.sqrt(self.alpha[t])
            * (
                xt
                - (self.beta[t])
                / torch.sqrt(1 - self.alpha_bar[t])
                * self.network(xt, t)
            )
        )
        std = torch.where(
            t > 0,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    def elbo(self, x0):
        """
        ELBO training objective + QMC sampler (VDM paper)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """
        if self.sampler == "simple":
            # Sample time step t using low-discrepancy sampling
            t = low_discrepancy_simple(x0.shape[0], self.T).to(x0.device)
        else:
            t = low_discrepancy_sobol(x0.shape[0], self.T).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        return -nn.MSELoss(reduction="mean")(epsilon, self.network(xt, t))

    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo(x0).mean()


#### IMPORTANCE SAMPLING ###############################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
history_length = 10  # Number of recent values to store
T = 1000
# Initialize a 2D tensor with size (T, history_length)
history = torch.zeros(T, history_length).to(device)
pointers = torch.zeros(T, dtype=torch.long).to(device)  # One pointer per type
weights = torch.ones(T).to(device) / history_length  # Weights for the moving average
n_updates = torch.zeros(T).to(device)  # Number of updates for each type


# Function to update the history tensor
def update_history(value, type_index):
    """
    Updates the history buffer for a specific type.

    Args:
    - value: The new value to add (float or tensor).
    - type_index: Index of the value type (0 <= type_index < T).
    """
    global pointers
    global n_updates
    pointer = pointers[type_index]  # Get the pointer for the type
    history[type_index, pointer] = value.unsqueeze(
        -1
    )  # Update the corresponding position
    pointers[type_index] = (pointer + 1) % history_length  # Move the pointer

    # count how many times each type_index has been updated
    counts = torch.bincount(type_index.squeeze(-1), minlength=T)
    n_updates += counts
    # print(n_updates)
    # compute the importance sampling weights

    # compute sqrt mean of history
    # print(history.shape)

    sqrt_mean_weights = torch.sqrt((history**2).mean(dim=1))

    weights_total = sqrt_mean_weights.sum()
    global weights
    weights = sqrt_mean_weights / weights_total


class DDPM_importance(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """

        super(DDPM_importance, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        mean = (
            1.0
            / torch.sqrt(self.alpha[t])
            * (
                xt
                - (self.beta[t])
                / torch.sqrt(1 - self.alpha_bar[t])
                * self.network(xt, t)
            )
        )
        std = torch.where(
            t > 0,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    def elbo(self, x0):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """

        # Sample time step t
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)
        elbo = -nn.MSELoss(reduction="none")(epsilon, self.network(xt, t)).mean(
            dim=1
        )  # observation-wise loss

        update_history(elbo, t)

        # if n_updates for all t are greater than 10, use the importance sampling weights
        scaling = 1
        have_printed = False
        if (n_updates[1:] > 10).all():
            # print once to notify
            scaling = 1 / weights[t].detach()
            # print(scaling)

        return elbo * scaling

    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo(x0).mean()


#### PREDICTING A DIFFERENT TARGET ################################################################################
#### DDPM for predicting mu #######################################################################################


class DDPM_mu(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2, pred="mu"):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """

        super(DDPM_mu, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T
        self.pred = pred
        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """
        if self.pred == "eps":
            mean = (
                1.0
                / torch.sqrt(self.alpha[t])
                * (
                    xt
                    - (self.beta[t])
                    / torch.sqrt(1 - self.alpha_bar[t])
                    * self.network(xt, t)
                )
            )
        elif self.pred == "mu":
            mean = self.network(xt, t)
        # torch.sqrt(((1-self.alpha_bar[t-1]) / (1-self.alpha_bar[t]))*
        std = torch.where(t > 0, torch.sqrt(self.beta[t]), 0)
        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    def elbo(self, x0):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """

        # Sample time step t
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        if self.pred == "eps":
            target = epsilon
            scaling = 1
        elif self.pred == "mu":
            target = (
                torch.sqrt(self.alpha_bar[t - 1])
                * self.beta[t]
                / (1 - self.alpha_bar[t])
            ) * x0 + (
                torch.sqrt(self.alpha[t])
                * (1 - self.alpha_bar[t - 1])
                / (1 - self.alpha_bar[t])
            ) * xt
            beta_tilde = (
                (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
            ) * self.beta[t]
            scaling = 1 / (
                2 * self.beta[t]
            )  # or a fixed variance if using an alternative variance schedule

        elbo = (
            -nn.MSELoss(reduction="none")(self.network(xt, t), target) * scaling
        ).mean(
            dim=1
        )  # observation-wise loss

        update_history(elbo, t)

        # if n_updates for all t are greater than 10, use the importance sampling weights
        is_scale = 1
        have_printed = False
        if (n_updates[1:] > 10).all():
            # print once to notify
            is_scale = 1 / weights[t].detach()
            # print(scaling)

        return elbo * is_scale

    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo(x0).mean()


#### DDPM for predicting x0 #######################################################################################


class DDPM_x0(nn.Module):
    def __init__(
        self,
        network,
        predict_using="x0",
        scheduler_method="linear",
        variance_reduction="none",
        T=100,
        beta_1=1e-4,
        beta_T=2e-2,
    ):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """

        super(DDPM_x0, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T

        # Predict using epsilon, mu or x_0
        self.predict_using = predict_using

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        # Eq 11 in Ho et al, 2020
        if self.predict_using == "epsilon":
            mean = (
                1.0
                / torch.sqrt(self.alpha[t])
                * (
                    xt
                    - (self.beta[t])
                    / torch.sqrt(1 - self.alpha_bar[t])
                    * self.network(xt, t)
                )
            )

            std = torch.where(
                t > 0,
                torch.sqrt(
                    ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t]))
                    * self.beta[t]
                ),
                0,
            )

        elif self.predict_using == "x0":
            nn_predicted_x0 = self.network(xt, t)  ## NN prediction of x0

            # Index alpha, alpha_bar, alpha_bar_pred,  beta
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev_t_1 = self.alpha_bar[t - 1]
            beta_t = self.beta[t]

            # Equation 7 in DDPM by Ho et al, 2020
            coefficient_x0 = torch.sqrt(alpha_bar_prev_t_1) / (1 - alpha_bar_t) * beta_t
            coefficient_xt = (
                torch.sqrt(alpha_t) * (1 - alpha_bar_prev_t_1) / (1 - alpha_bar_t)
            )
            mean = coefficient_x0 * nn_predicted_x0 + coefficient_xt * xt

            beta_tilde = ((1 - alpha_bar_prev_t_1) / (1 - alpha_bar_t)) * beta_t
            std = torch.sqrt(beta_tilde)

        elif self.predict_using == "mu":
            mean = self.network(xt, t)

            std = torch.where(
                t > 0,
                torch.sqrt(
                    ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t]))
                    * self.beta[t]
                ),
                0,
            )

        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    def elbo(self, x0):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """

        # Sample time step t
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        if self.predict_using == "epsilon":
            target = epsilon
        elif self.predict_using == "x0":
            target = x0
        elif self.predict_using == "mu":
            target = self.forward_diffusion(x0, t - 1, epsilon)

        return -nn.MSELoss(reduction="none")(target, self.network(xt, t))

    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """

        return -self.elbo(x0).mean()


#### CLASSIFIER GUIDANCE DIFFUSION #######################################################################################


class DDPM_class(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """

        super(DDPM_class, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, y, t, epsilon, w=1.0, classifier=None):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """
        with torch.enable_grad():  # Ensure gradient tracking is enabled
            xt.requires_grad_(True)  # Enable gradients for xt
            grad = classifier.compute_log_gradients(xt, target_class=y)
        eps_theta = self.network(xt, t) - torch.sqrt(1 - self.alpha_bar[t]) * grad * w

        mean = (
            1.0
            / torch.sqrt(self.alpha[t])
            * (xt - (self.beta[t]) / torch.sqrt(1 - self.alpha_bar[t]) * eps_theta)
        )
        std = torch.where(
            t > 0,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * epsilon

    # @torch.no_grad()
    def sample(self, shape, y, w=1.0, classifier=None):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)
        # y = torch.arange(shape[0]).to(self.beta.device)
        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)

            xt = self.reverse_diffusion(xt, y, t, noise, w, classifier)

        return xt

    def elbo(self, x0):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """

        # Sample time step t
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        return -nn.MSELoss(reduction="mean")(epsilon, self.network(xt, t))

    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo(x0).mean()


#### CLASSIFIER-FREE GUIDANCE DIFFUSION #######################################################################################


def set_non_guidance_labels(classes, num_classes=10, non_guidance_index=10, p=0.1):
    """
    Randomly sets a portion of class labels to a non-guidance index.

    Parameters
    ----------
    classes : torch.Tensor
        Tensor of class labels with values in the range [0, num_classes - 1].
    num_classes : int, optional
        Total number of classes (default is 10 for MNIST).
    non_guidance_index : int, optional
        Index representing the non-guidance class (default is 10).
    p : float, optional
        Probability of replacing a label with the non-guidance index (default is 0.1).

    Returns
    -------
    torch.Tensor
        Modified class labels with approximately `p` proportion set to the non-guidance index.
    """
    batch_size = classes.size(0)
    # Generate a mask with `p` probability
    mask = torch.rand(batch_size) < p
    # Replace the selected indices with the non-guidance index
    classes[mask] = non_guidance_index
    return classes


class DDPM_class_free(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """

        super(DDPM_class_free, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t, y: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T), y)
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, y, t, epsilon, w=1):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """
        uncod_class = torch.full_like(y, 10).to(self.beta.device)

        eps = (1 + w) * self.network(xt, t, uncod_class) - w * self.network(xt, t, y)
        mean = (
            1.0
            / torch.sqrt(self.alpha[t])
            * (xt - (self.beta[t]) / torch.sqrt(1 - self.alpha_bar[t]) * eps)
        )
        std = torch.where(
            t > 0,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape, w=1):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)
        y = torch.arange(shape[0]).to(self.beta.device)
        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)

            xt = self.reverse_diffusion(xt, y, t, noise, w)

        return xt

    @torch.no_grad()
    def sample_guided(self, shape, y, w=1):
        """
        Sample from the diffusion model following the guidance

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)
        y : tensor
            Classes to sample
        w : int
            Strenght of the guidance

        Returns
        -------
        torch.tensor
            sampled image
        """
        y = y.to(self.beta.device)

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)
        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)

            xt = self.reverse_diffusion(xt, y, t, noise, w)

        return xt

    def elbo(self, x0, y):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """

        # Sample time step t
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        y_with_non_guide = set_non_guidance_labels(
            y, num_classes=10, non_guidance_index=10, p=0.1
        )  # set random labels to non-guidance index

        return -nn.MSELoss(reduction="mean")(
            epsilon, self.network(xt, t, y_with_non_guide)
        )

    def loss(self, x0, y):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo(x0, y).mean()
