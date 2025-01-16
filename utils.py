import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
import os
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from torcheval.metrics import FrechetInceptionDistance
from tqdm.auto import tqdm
from scipy.stats import qmc



def reporter(model):
    """Callback function used for plotting images during training"""

    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        nsamples = 10
        samples = model.sample((nsamples, 28 * 28)).cpu()

        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)

        # Plot in grid
        grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.show()


def train(model, optimizer, scheduler, dataloader, epochs, device, ema=True, per_epoch_callback=None):
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
        ema_model = ExponentialMovingAverage(model, device=device, decay=1.0 - ema_alpha)

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
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch + 1}/{epochs}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2E}")
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter % ema_steps == 0:
                    ema_model.update_parameters(model)

        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model)


# printing samples once model is trained
def get_samples(model, n_samples):
    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        nsamples = n_samples
        samples = model.sample((nsamples, 28 * 28)).cpu()

        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)

        # Plot in grid
        grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.show()

def generate_save_samples(model,
                          mnist_dataloader,
                          root_dir="./generated_images",
                          n_samples=10000,
                          batch_size=256,
                          guided=False,
                          w=None,
                          plot=False):
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

    progress_bar = tqdm(range(n_samples), desc="Image Generation")

    # root directory
    root_dir = root_dir
    # class directory (for dataloader)
    class_name = "generated"
    class_dir = os.path.join(root_dir, class_name)
    # create the directory
    os.makedirs(class_dir, exist_ok=True)

    # Generate and save images
    # model in evaluation mode
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(mnist_dataloader):
            if batch_idx * batch_size >= n_samples:
                break
            batch_size_curr = min(batch_size, n_samples - batch_idx * batch_size)
            if guided:
                # taking the labels
                y = data[1]
                y = y[:batch_size]  # Ensure the label batch size matches
                samples = model.sample((batch_size_curr, 28 * 28), y, w)
            else:
                samples = model.sample((batch_size_curr, 28 * 28))

            samples = (samples + 1) / 2  # Map back from [-1, 1] to [0, 1]
            samples = samples.clamp(0.0, 1.0)

            for j, sample in enumerate(samples):
                global_index = batch_idx * batch_size + j  # Global index of the sample
                save_path = os.path.join(class_dir, f'image_{global_index:05d}.png')
                save_image(sample.view(1, 28, 28), save_path)

                # Update progress bar
                progress_bar.set_postfix(image=f"⠀{global_index:5d}")
                progress_bar.update()

                # Visualize every 100th image
                if global_index % 100 == 0:
                    print(f'Generated {global_index} images')
                    if plot:
                        print(f"Visualizing image at index: {global_index}")
                        plt.imshow(sample.view(28, 28).cpu(), cmap="gray")
                        plt.axis("off")
                        plt.show()


def compute_fid(generated_images_dir="./generated_images",
                evaluation_images_dir="./evaluation_images",
                train_mnist=False,
                download_mnist=True,
                batch_size=256,
                device="cpu",
                eval_batches=None,
                feature_dim=2048):
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
    Returns
    -------
    float
        Computed FID value.
    """
    if not train_mnist:
        print("Evaluation on MNIST test set...")
    else:
        print("Evaluation on MNIST training set...")

    # Image transformation (both evaluation set and generated one)
    transform_fid = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ensuring 3 channels for FID obj.
        transforms.ToTensor(),
    ])

    # Generated sample manipulation
    # Create a dataset from the folder
    dataset = datasets.ImageFolder(generated_images_dir,
                                   transform=transform_fid)

    # Create a DataLoader from the dataset
    dataloader_gen = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

    # Real data manipulation
    dataloader_eval = torch.utils.data.DataLoader(
        datasets.MNIST(evaluation_images_dir,
                       download=download_mnist,
                       train=train_mnist,
                       transform=transform_fid),
        batch_size=batch_size,
        shuffle=True)

    # Initializing the FID object
    fid = FrechetInceptionDistance(
        model=None,  # use default model for feature activation
        device=device,
        feature_dim=feature_dim
    )

    if eval_batches:
        print(f"Using only {eval_batches} batches.")
        progress = eval_batches
    else:
        progress = math.ceil(10000 / batch_size)

    progress_bar1 = tqdm(range(progress), desc="Loading real data into FID object")
    progress_bar2 = tqdm(range(progress), desc="Loading generated data into FID object")

    # Loading real images to FID object
    for batch_idx, (data, _) in enumerate(dataloader_eval):
        fid.update(data, is_real=True)
        progress_bar1.set_postfix(batch=f"⠀{batch_idx:5d}")
        progress_bar1.update()
        if eval_batches is not None and batch_idx == eval_batches - 1:
            break

    # Loading generated images to FID object
    for batch_idx, (data, _) in enumerate(dataloader_gen):
        fid.update(data, is_real=False)
        progress_bar2.set_postfix(batch=f"⠀{batch_idx:5d}")
        progress_bar2.update()
        if eval_batches is not None and batch_idx == eval_batches - 1:
            break

    print("Computing FID...")
    res = fid.compute()
    print(f"FID: {res}")

    return res