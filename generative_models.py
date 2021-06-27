"""
This module implements deep generative models in pytorch.

Notes:
    No input channels are considered. 
    Considered DGMs use MLPs with Batch Normalization.
"""

import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
import pytorch_lightning as pl

from functools import partial
import random

def block(in_feat, out_feat, act_fun, act_args):
    """
    Implements an MLP's layer with Batch Normalization 1D.
    
    Args:
        in_feat: scalar int.
        out_feat: scalar int.
        act_fun: torch.nn.Module activation function.
        act_args: list of arguments for act_fun
    Returns:
        layer: a layer for an MLP
    """
    layer = [nn.Linear(in_feat, out_feat)]
    layer.append(nn.BatchNorm1d(out_feat))
    layer.append(act_fun(*act_args))
    return layer

def MLP_layers(hidden_dims, in_dim, act_fun, act_args):
    """
    Produces a list of model layers.
    
    Args:
        in_dim: scalar int. 
        hidden_dims: scalar int list. 
            Number of hidden dimensions for each layer. 
        act_fun: torch.nn.Module.
            Activation function for all intermediate layers.
        act_args: list of arguments for act_fun
    Returns: 
        layers: list of layers for an MLP.
    """
    layers = []
    d_in = in_dim
    for h_dim in hidden_dims:
        layers = layers + block(d_in, h_dim, act_fun, act_args)
        d_in = h_dim
    return layers


class Generator(nn.Module):
    """
    Implements an MLP. This is used as the generator of different 
    Deep Generative Models.
    """
    def __init__(self, hidden_dims, latent_dim, ambient_dim, act_fun, act_args):
        super(Generator, self).__init__()
        """
        Constructor for a Generator network.
        
        Args:
            latent_dim: scalar int. 
                Dimensionality of the latent space.
            ambient_dim: scalar int. 
                Dimensionality of the ambient space.
            hidden_dims: scalar int list. 
                Number of hidden dimensions for each layer. 
            act_fun: torch.nn.Module.
                Activation function for all intermediate layers.
            act_args: list of arguments for act_fun
        """
        self.latent_dim = latent_dim

        layers = MLP_layers(hidden_dims, latent_dim, act_fun, act_args)
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], ambient_dim))
        layers.append(nn.Sigmoid()) # One channel

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        """
        Generate a sample on the ambient space corresponding to some input
        on the latent space. 
        """
        sample = self.model(input.view(-1, self.latent_dim))
        return sample

class VAE(pl.LightningModule):
    """
    Implements a Variational AutoEncoder Kingma & Welling (2013). 
    """
    def __init__(self, hidden_dims, latent_dim, ambient_dim, 
                act_fun, act_args, optimizer_class, optimizer_kwargs):
        """
        Constructor for a Variational AutoEncoder.
        Hidden dimensions are shared between the encoder and decoder. 
        The constructor assumes that they are ordered from ambient_dim to latent_dim.
        
        Args:
            latent_dim: scalar int. 
                Dimensionality of the latent space.
            ambient_dim: scalar int. 
                Dimensionality of the ambient space.
            hidden_dims: scalar int list. 
                Number of hidden dimensions for each layer. 
            act_fun: torch.nn.Module.
                Activation function for all intermediate layers.
            act_args: list of arguments for act_fun
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.ambient_dim = ambient_dim
        self.optimizer_class = partial(optimizer_class, **optimizer_kwargs)
        
        # Encoder
        enc_layers = MLP_layers(hidden_dims, ambient_dim, act_fun, act_args)
        self.encoder = nn.Sequential(*enc_layers)
        
        # Mu and sigma layers
        self.mu_layer = nn.Linear(hidden_dims[-1],latent_dim)
        self.var_layer = nn.Linear(hidden_dims[-1],latent_dim)
        
        # Decoder (Generator)
        self.decode = Generator(
            # hidden_dims for the encoder, their reverse for the decoder.
            hidden_dims[::-1], latent_dim, ambient_dim, act_fun, act_args
            )
        

    def encode(self, sample):
        """
        Encodes a data sample.
        
        Args:
            sample: tensor of (batch_size, ambient_dim)
        Returns
            tensor outputs of the mean and variance networks.
        """
        hid_state = self.encoder(sample)
        return self.mu_layer(hid_state), self.var_layer(hid_state)

    def reparameterize(self, mu, logvar):
        """
        Produces a random latent representation by applying the 
        reparametrization trick to the mean and variance of the encoding.
        
        Args:
            mu: tensor of (batch_size, latent_dim)
                means of the encoding.
            logvar: tensor of (batch_size, latent_dim)
                log variance per dimension of the encoding.
        Returns
            tensor latent representation of encoded data.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, sample):
        """
        Encodes and decodes a sample.
        
        Args:
            sample: tensor of (batch_size, ambient_dim)
        Returns
            Reconstruction: tensor.
            Encoded means: tensor.
            Encoded log-variances: tensor.
        """
        mu, logvar = self.encode(sample)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def configure_optimizers(self):
        """
        Configures the optimizer for the VAEs parameters.
        """
        model_opt = self.optimizer_class(self.parameters())
        return model_opt
    
    def eval_loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def training_step(self, batch, batch_idx):
        """
        Training step for a Variational AutoEncoder
        
        Args:
            batch: tensor of (batch_size, ambient_dim), labels of (batch_size, 1)
            
        Returns:
            VAE loss
        """
        sample, _ = batch
        sample = sample.view(-1, self.ambient_dim)
        recon_sample, mu, logvar = self(sample)
        
        loss = self.eval_loss(recon_sample, sample, mu, logvar)
        self.log_dict({"Loss:":loss})
        
        return loss

class DAE(pl.LightningModule):
    """
    Implements a Denoising AutoEncoder.  
    """
    def __init__(self, hidden_dims, latent_dim, ambient_dim, 
                act_fun, act_args, optimizer_class, optimizer_kwargs,
                gaussian_std=0.1, speckle_std=0.1):
        """
        Constructor for a Denoising AutoEncoder.
        Hidden dimensions are shared between the encoder and decoder. The 
        constructor assumes that they are ordered from ambient_dim to latent_dim.
        Gaussian and speckle noise are considered for sample perturbations.
        
        Args:
            latent_dim: scalar int. 
                Dimensionality of the latent space.
            ambient_dim: scalar int. 
                Dimensionality of the ambient space.
            hidden_dims: scalar int list. 
                Number of hidden dimensions for each layer. 
            act_fun: torch.nn.Module.
                Activation function for all intermediate layers.
            act_args: list of arguments for act_fun
            gaussian_std: scalar float:
                standard deviation of the sample gaussian noise
            speckle_std: scalar float:
                standard deviation of the sample speckle noise
        """
        super(DAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.ambient_dim = ambient_dim
        self.optimizer_class = partial(optimizer_class, **optimizer_kwargs)
        
        # Noise functions
        self.gaussian_std=gaussian_std
        self.speckle_std=speckle_std
        
        # Encoder
        enc_layers = MLP_layers(hidden_dims, ambient_dim, act_fun, act_args)
        enc_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encode = nn.Sequential(*enc_layers)

        # Decoder (Generator)
        self.decode = Generator(
            # hidden_dims for the encoder, their reverse for the decoder.
            hidden_dims[::-1], latent_dim, ambient_dim, act_fun, act_args
            )
        
        
    def forward(self, sample):
        """
        Encodes and decodes a sample.
        """
        # Add noise
        t=random.choice([self.add_speckle,self.add_gaussian])
        sample = t(sample)
        
        return self.decode(self.encode(sample))
    
    def configure_optimizers(self):
        """
        Configures the optimizer for the VAEs parameters.
        """
        model_opt = self.optimizer_class(self.parameters())
        return model_opt

    def training_step(self, batch, batch_idx):
        """
        Training step for a Variational AutoEncoder

        Args:
            batch: tensor of (batch_size, ambient_dim), labels of (batch_size, 1)

        Returns:
            VAE loss
        """
        sample, _ = batch
        sample = sample.view(-1, self.ambient_dim)
        recon_batch = self(sample)
        
        loss = F.binary_cross_entropy(recon_batch, sample, reduction = "sum")
        self.log_dict({"Loss:":loss})
        
        return loss
    
    def add_gaussian(self, tensor):
        noise = torch.randn(tensor.size()).to(self.device)
        return tensor + noise * self.gaussian_std
    def add_speckle(self, tensor):
        noise = torch.randn(tensor.size()).to(self.device)
        return tensor + noise * tensor * self.speckle_std
    
    
class GAN(pl.LightningModule):
    """
    Implements a Generative Adversarial Network (Goodfellow et al., 2014).  
    """
    def __init__(self, hidden_dims, latent_dim, ambient_dim, 
                act_fun, act_args, optimizer_class, optimizer_kwargs):
        """
        Constructor for a Generative Adversarial Network.
        Hidden dimensions are shared between the generator and disctiminator. The 
        constructor assumes that they are ordered from ambient_dim to latent_dim.
        
        Args:
            latent_dim: scalar int. 
                Dimensionality of the latent space.
            ambient_dim: scalar int. 
                Dimensionality of the ambient space.
            hidden_dims: scalar int list. 
                Number of hidden dimensions for each layer. 
            act_fun: torch.nn.Module.
                Activation function for all intermediate layers.
            act_args: list of arguments for act_fun
        """
        super(GAN, self).__init__()
        
        self.latent_dim = latent_dim
        self.ambient_dim = ambient_dim
        self.optimizer_class = partial(optimizer_class, **optimizer_kwargs)
        self.criterion = F.binary_cross_entropy # do not hard code
        
        # Discriminator
        dis_layers = MLP_layers(
            hidden_dims, ambient_dim, 
            nn.LeakyReLU, (0.2, True) # better for a discriminator
            )
        dis_layers += [nn.Linear(hidden_dims[-1], 1), nn.Sigmoid()]
        self.discriminator = nn.Sequential(*dis_layers)
        
        # Generator
        self.generator = Generator(
            # hidden_dims for the discriminator, their reverse for the decoder.
            hidden_dims[::-1], latent_dim, ambient_dim, act_fun, act_args
            )
        
    def forward(self, z):
        """
        Generates a sample from a latent space represention.    
        """
        return self.generator(z)
    
    def configure_optimizers(self):
        """
        Configures the optimizers for the Generator and Discriminator.
        """
        optimizer_D = self.optimizer_class(self.discriminator.parameters())
        optimizer_G = self.optimizer_class(self.generator.parameters())
        return optimizer_G, optimizer_D

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Training step for a GAN

        Args:
            batch: tensor of (batch_size, ambient_dim), labels of (batch_size, 1)
            optimizer_idx: int
                one of [0,1] representing whether the step is on the 
                discriminator or the generator
        Returns:
            Generator loss for optimizer_G or 
            Discriminator loss for optimizer_D
        """
        # batch returns x and y tensors
        real_sample, _ = batch
        real_sample = real_sample.view(-1, self.ambient_dim)
        
        # ground truth (tensors of ones and zeros) same shape as images
        valid = torch.ones(real_sample.size(0), 1).to(self.device)
        fake = torch.zeros(real_sample.size(0), 1).to(self.device)
        
        if optimizer_idx == 0:
            # Generator
            gen_input = torch.randn(real_sample.size(0), self.latent_dim) # noise 
            gen_input = gen_input.to(self.device)           
            self.gen_sample = self.generator(gen_input)
            
            # Does it fool the discriminator?
            g_loss = self.criterion(self.discriminator(self.gen_sample), valid)
            self.log_dict({"Generator loss:":g_loss})
            
            return g_loss
        
        if optimizer_idx == 1:
            # Discriminator
            
            real_loss = self.criterion(self.discriminator(real_sample), valid) 
            fake_loss = self.criterion(
                self.discriminator(self.gen_sample.detach()), 
                fake
                )
            d_loss = (real_loss + fake_loss)/2.0
            self.log_dict({"Discriminator loss:":d_loss})
            
            return d_loss