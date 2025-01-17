import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from torcheval.metrics import FrechetInceptionDistance
from tqdm.auto import tqdm


#### U-net definitions #################################################################################################
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class ScoreNet2(nn.Module):
    # A time-dependent score-based model with conditional mean computation.

    def __init__(self, channels=[32, 64, 128, 256], embed_dim=256, group_num=4):

        super().__init__()

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(group_num, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(group_num, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(group_num, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(group_num, num_channels=channels[3])
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(group_num, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(group_num, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(group_num, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

        # Parameters for conditional mean computation
        self.rho0 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.rho1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x, t):

        embed = self.act(self.embed(t))

        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Compute conditional mean μ(x, t; θ)
        mean = self.rho0 * (x - self.rho1 * h)

        return mean


class ScoreNet_class(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        marginal_prob_std,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        num_classes=10 + 1,
    ):  # 10 classes for digits, 1 for wildcard
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.class_embed = nn.Embedding(num_classes, embed_dim)  # class embedding
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, y):
        # Obtain the Gaussian random feature embedding for t
        time_embed = self.act(self.embed(t))
        class_embed = self.act(self.class_embed(y))
        embed = time_embed + class_embed
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


#### Classifier definitions #################################################################################################


class RobustMNISTClassifier(nn.Module):
    def __init__(self):
        super(RobustMNISTClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # Output: (32, 28, 28)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1
        )  # Output: (64, 28, 28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 14, 14)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Apply convolutions with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten for fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Apply fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here (logits)

        return x


class ClassifierWrapper(nn.Module):
    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        super(ClassifierWrapper, self).__init__()

        self._network = network
        self.network = lambda x: (
            self._network(x.reshape(-1, 1, 28, 28)).reshape(-1, 10)
        )

        self.criterion = nn.CrossEntropyLoss()

        self.T = T

        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def loss(self, x0, y):
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)
        epsilon = torch.randn_like(x0)
        xt = self.forward_diffusion(x0, t, epsilon)
        return self.criterion(self.network(xt), y)

    def forward_diffusion(self, x0, t, epsilon):
        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])
        return mean + std * epsilon

    def compute_log_gradients(self, xt, target_class):
        xt.requires_grad_(True)
        logits = self.network(xt)
        log_probs = torch.log_softmax(logits, dim=1)
        target_log_probs = log_probs[torch.arange(log_probs.size(0)), target_class]
        gradients = torch.autograd.grad(outputs=target_log_probs.sum(), inputs=xt)[0]
        return gradients


#### Support Functions #################################################################################################
def reporter(model):
    """Callback function used for plotting images during training"""

    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        nsamples = 10
        # setting random seed fro comparability of results
        torch.manual_seed(42)
        samples = model.sample((nsamples, 28 * 28)).cpu()

        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)

        # Plot in grid
        grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.show()


def train(
    model,
    optimizer,
    scheduler,
    dataloader,
    epochs,
    device,
    ema=True,
    per_epoch_callback=None,
):
    """
    Training loop

    Parameters
    ----------
    model: nn.Module
        Pytorch model
    optimizer: optim.Optimizer
        Pytorch optimizer to be used for training
    scheduler: optim.LRScheduler
        Pytorch learning rate scheduler
    dataloader: utils.DataLoader
        Pytorch dataloader
    epochs: int
        Number of epochs to train
    device: torch.device
        Pytorch device specification
    ema: Boolean
        Whether to activate Exponential Model Averaging
    per_epoch_callback: function
        Called at the end of every epoch
    """

    # Setup progress bar
    total_steps = len(dataloader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(
            model, device=device, decay=1.0 - ema_alpha
        )

    for epoch in range(epochs):

        # Switch to train mode
        model.train()

        global_step_counter = 0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}",
                epoch=f"{epoch + 1}/{epochs}",
                lr=f"{scheduler.get_last_lr()[0]:.2E}",
            )
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter % ema_steps == 0:
                    ema_model.update_parameters(model)

        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model)


def train_class_free(
    model,
    optimizer,
    scheduler,
    dataloader,
    epochs,
    device,
    ema=True,
    per_epoch_callback=None,
):
    """
    Training loop for classifier-free guidance model

    Parameters
    ----------
    model: nn.Module
        Pytorch model
    optimizer: optim.Optimizer
        Pytorch optimizer to be used for training
    scheduler: optim.LRScheduler
        Pytorch learning rate scheduler
    dataloader: utils.DataLoader
        Pytorch dataloader
    epochs: int
        Number of epochs to train
    device: torch.device
        Pytorch device specification
    ema: Boolean
        Whether to activate Exponential Model Averaging
    per_epoch_callback: function
        Called at the end of every epoch
    """

    # Setup progress bar
    total_steps = len(dataloader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(
            model, device=device, decay=1.0 - ema_alpha
        )

    for epoch in range(epochs):

        # Switch to train mode
        model.train()

        global_step_counter = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}",
                lr=f"{scheduler.get_last_lr()[0]:.2E}",
            )
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter % ema_steps == 0:
                    ema_model.update_parameters(model)


def train_classifier(model_classifier, wrapper):

    learning_rate = 1e-3
    epochs = 4
    batch_size = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST("./mnist_data", download=True, train=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )

    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./mnist_data", download=True, train=False, transform=test_transform
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model_classifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
    progress_bar = tqdm(range(len(dataloader) * epochs), desc="Training")

    for epoch in range(epochs):
        model_classifier.train()
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = wrapper.loss(x, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                epoch=f"{epoch+1}/{epochs}",
                lr=f"{scheduler.get_last_lr()[0]:.2E}",
            )
            progress_bar.update()

    # Evaluation
    model_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model_classifier(x.reshape(-1, 1, 28, 28))
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Classifier Accuracy: {accuracy:.2f}%")

    return wrapper


# printing samples once model is trained
def get_samples(model, n_samples, seed=42):
    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        nsamples = n_samples
        torch.manual_seed(42)
        samples = model.sample((nsamples, 28 * 28))

        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)

        # Plot in grid
        grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.show()


#### Model Comparison #################################################################################################
def generate_save_samples(
    model,
    mnist_dataloader,
    root_dir="./generated_images",
    n_samples=10000,
    batch_size=256,
    guided=False,
    w=None,
    classifier=None,
    plot=False,
):
    """
    Generate and save images from the model.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    mnist_dataloader : DataLoader
        DataLoader for the MNIST dataset.
    root_dir : str
        Root directory to save the generated images. Default is "./generated_images".
    n_samples : int
        Number of samples to generate. Default is 10000.
    batch_size : int
        Batch size for generating samples. Default is 256.
    guided : bool
        Whether to generate guided samples. Default is False.
    w   : int/float
        hyperparameter for the strength of the guided samples. Default is None.
    plot : bool
        Whether to plot the images. Default is False.

    Returns
    -------
    None
    """

    # root directory
    root_dir = root_dir
    # class directory (for dataloader)
    class_name = "generated"
    class_dir = os.path.join(root_dir, class_name)

    # create the directory
    if os.path.exists(class_dir) and os.path.isdir(class_dir):
        print("Sample have been already generated.\n")
        return
    else:
        os.makedirs(class_dir, exist_ok=False)

    # Generate and save images
    # model in evaluation mode
    model.eval()
    progress_bar = tqdm(range(n_samples), desc="Image Generation")

    with torch.no_grad():
        for batch_idx, data in enumerate(mnist_dataloader):
            if batch_idx * batch_size >= n_samples:
                break
            batch_size_curr = min(batch_size, n_samples - batch_idx * batch_size)
            if guided:
                # taking the labels
                y = data[1]
                y = y[:batch_size]  # Ensure the label batch size matches
                if classifier is not None:
                    samples = model.sample(
                        (batch_size_curr, 28 * 28), y, w, classifier=classifier
                    )
                else:
                    samples = model.sample_guided((batch_size_curr, 28 * 28), y, w)
            else:
                samples = model.sample((batch_size_curr, 28 * 28))

            samples = (samples + 1) / 2  # Map back from [-1, 1] to [0, 1]
            samples = samples.clamp(0.0, 1.0)

            for j, sample in enumerate(samples):
                global_index = batch_idx * batch_size + j  # Global index of the sample
                save_path = os.path.join(class_dir, f"image_{global_index:05d}.png")
                save_image(sample.view(1, 28, 28), save_path)

                # Update progress bar
                progress_bar.set_postfix(image=f"{global_index+1:5d}")
                progress_bar.update()

                # Visualize every 100th image
                if global_index % 100 == 0:
                    if plot:
                        print(f"Visualizing image at index: {global_index}")
                        plt.imshow(sample.view(28, 28).cpu(), cmap="gray")
                        plt.axis("off")
                        plt.show()


def compute_fid(
    generated_images_dir="./generated_images",
    evaluation_images_dir="./evaluation_images",
    train_mnist=False,
    download_mnist=False,
    batch_size=256,
    shuffle=False,
    device="cpu",
    eval_batches=None,
    feature_dim=2048,
    seed=42,
):
    """
    Compute the Frechet Inception Distance (FID) between the generated images and the evaluation images.

    Parameters
    ----------
    generated_images_dir : str
        Directory containing the generated images. Default is "./generated_images".
    evaluation_images_dir : str
        Directory containing the evaluation images. Default is "./evaluation_images".
    train_mnist : bool
        Whether to use the training set of MNIST. Default is False.
    download_mnist  : bool
        Whether to download the MNIST dataset. Default is True.
    batch_size  : int
        Batch size for computing FID. Default is 256.
    device  : str
        Device to run computations on (CPU or GPU). Default is "cpu".
    eval_batches : int
        Number of batches to evaluate. Default is None.
    feature_dim : int
        Dimensionality of the feature space for Inception V3 model. Possible values are 64, 192, 768, or 2048. Default is 2048.
    seed : int
        Random seed
    Returns
    -------
    float
        Computed FID value.
    """
    torch.manual_seed(seed)

    if not train_mnist:
        print("Evaluation on MNIST test set...")
    else:
        print("Evaluation on MNIST training set...")

    # Image transformation (both evaluation set and generated one)
    transform_fid = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),  # ensuring 3 channels for FID obj.
            transforms.ToTensor(),
        ]
    )

    # Generated sample manipulation
    # Create a dataset from the folder
    dataset = datasets.ImageFolder(generated_images_dir, transform=transform_fid)

    # Create a DataLoader from the dataset
    dataloader_gen = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    # Real data manipulation
    dataloader_eval = torch.utils.data.DataLoader(
        datasets.MNIST(
            evaluation_images_dir,
            download=download_mnist,
            train=train_mnist,
            transform=transform_fid,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Initializing the FID object
    fid = FrechetInceptionDistance(
        model=None,  # use default model for feature activation
        device=device,
        feature_dim=feature_dim,
    )

    if eval_batches:
        print(f"Using only {eval_batches} batches.")
        progress = eval_batches
    else:
        progress = math.ceil(10000 / batch_size)

    progress_bar1 = tqdm(range(progress), desc="Loading real data into FID object")
    # Loading real images to FID object
    for batch_idx, (data, _) in enumerate(dataloader_eval):
        fid.update(data, is_real=True)
        progress_bar1.set_postfix(batch=f"{batch_idx+1:3d}")
        progress_bar1.update()
        if eval_batches is not None and batch_idx == eval_batches - 1:
            break

    progress_bar2 = tqdm(range(progress), desc="Loading generated data into FID object")
    # Loading generated images to FID object
    for batch_idx, (data, _) in enumerate(dataloader_gen):
        fid.update(data, is_real=False)
        progress_bar2.set_postfix(batch=f"{batch_idx+1:3d}")
        progress_bar2.update()
        if eval_batches is not None and batch_idx == eval_batches - 1:
            break

    print("Computing FID...")
    res = fid.compute()
    print(f"FID: {res}.\n")

    return res
