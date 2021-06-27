"""
This module implements some visualizations of Images, their latent
space representations and the jacobian of their generator network.
"""

import dataset_utils

import numpy as np
import torch
from torch.autograd.functional import jacobian

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

def manifold_limits(model, data_size=60000):
    """
    Find the bounds of the data manifold on the latent space.
    
    Args:
        model: torch.nn.Module 
            A class with a defined encoder method
        data_size: int, optional
            amount of data points to consider
    """
    name = get_name(model) # VAE, DAE or GAN
    
    # Get all the data points. This is fine for our usecase
    mini_loader= dataset_utils.create_loader(data_size,data_size, list(range(10))) #hardcoded
    
    for (imgs,_) in mini_loader:
        z = model.encode(imgs.view(imgs.size(0),-1).to(model.device))
        if name == "VAE":
            # only mean outputs
            z = z[0]
        z = z.view(-1,2).detach().cpu()
    
    # Bound the provided grid
    xmin,xmax=np.percentile(z[:,0],1),np.percentile(z[:,0],99)
    ymin,ymax=np.percentile(z[:,1],1),np.percentile(z[:,1],99)
    
    return xmin,xmax,ymin,ymax

##############################
# Show images
##############################

def plot_generated(model, num_real=200):
    """
    Show a generated image from a model. A vector is sampled from a 
    standard multivariate Gaussian distribution and passed through the
    generator of the model. In principle, only works for VAEs and GANs.
    
    Also, the point's location on the latent space is shown.
    
    Args:
        model: torch.nn.Module 
            A VAE or GAN.
        num_real (int, optional): number of real datapoints to show on the 
        latent space of the model as reference. Defaults to 200.
    """
    # Get the generator
    if get_name(model) == "GAN":
        generator=model.generator
    else:
        generator=model.decode
    
    z=torch.randn(1,2).to(model.device)
    x=generator(z).detach().cpu().view(28,28)
    
    # Show the generated image
    image=plt.imshow(x,cmap="gray")
    
    # Show the latent space, and where the image came from
    if model.encode != None:
        latent = plot_latent_space(model, num_real, list(range(10)))
    else: latent = plt.figure(figsize=(5, 4))
    
    z=z.cpu()
    latent.scatter(z[0,0],z[0,1],c="black",marker='*',s=100,label="Gen")
    latent.legend()
    
    return image

def show_rec(model):
    """
    Plot a random image and its reconstruction. Only works with
    AutoEncoders.
    
    Args:
        model: torch.nn.Module 
            An autoencoder with defined encoder and decoder methods.
    """
    # Sample one random datapoint. Hard coded
    mini_loader = dataset_utils.create_loader(
        size=60000,
        batch_size=1
        )
    
    # Do a grid for the two imgs: the sample and its reconstruction
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(3,1))
    gs1 = gridspec.GridSpec(1,2)
    gs1.update(wspace=0.005, hspace=0.005) 
    
    for img,_ in mini_loader:
        img=img[0].reshape(28,28) # format
    
    # Original Image
    ax1.imshow(img,cmap='gray')
    ax1.set_title("Original Image")
    
    # Show reconstruction
    name = get_name(model) # VAE, DAE or GAN
    img=img.to(model.device).view(1,-1)
    code=model.encode(img)
    if name == "vae":
        code=code[0]
    recons=model.decode(code) # Reconstruction
    ax2.imshow(recons.view(28,28).detach().cpu(),cmap='gray')
    ax2.set_title("Reconstruction")
    
    # remove the x and y ticks
    for ax in (ax1,ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    
    return fig

def plot_image_grid(model, hor_ims=5, vert_ims=5, plot_size=(9,10),
                    im_h=28, im_w=28, xmin=-1, xmax=1, ymin=-1, ymax=1):
    """
    Plot images produced from different points on the latent space of 
    a generative model. These points come from an equidistant division
    of the space.
    Args:
        model: torch.nn.Module
            A [deep] genertive model
        hor_ims (int, optional): Number of horizontal images to show. 
            Defaults to 5.
        vert_ims (int, optional): Number of vertical images to show. 
            Defaults to 5.
        plot_size (tuple, optional): Defaults to (9,10).
        im_h (int, optional): Number of pixels (height) an image has. 
            Defaults to 28.
        im_w (int, optional): Number of pixels (width) an image has.
            Defaults to 28.
        xmin (int, optional): Grid limit. Defaults to -1.
        xmax (int, optional): Grid limit. Defaults to 1.
        ymin (int, optional): Grid limit. Defaults to -1.
        ymax (int, optional): Grid limit. Defaults to 1.
    
    """
    
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(111)

    # Grid of the latent space
    z1,z2=torch.linspace(xmin,xmax,hor_ims),torch.linspace(ymin,ymax,vert_ims)
    z = torch.cartesian_prod(z1, z2).to(model.device) 
    
    # Get the generator
    if get_name(model) == "GAN":
        generator=model.generator
    else:
        generator=model.decode
    
    # Generate images
    data_tensor = generator(z).detach().cpu()
    
    reshaped_tensor = np.zeros((int(im_h * vert_ims), int(im_w * hor_ims)))
    for row in range(vert_ims):
        for col in range(hor_ims):
            # Delimit the space this image takes up
            col_inf, col_sup = (int(col*im_w), int((col+1)*im_w))
            row_inf, row_sup = (int(row*im_w), int((row+1)*im_w))
            # Reshape image to fit
            reshaped_im = np.reshape(data_tensor[int(col + hor_ims * row), :], (im_h, im_w))
            reshaped_tensor[row_inf:row_sup, col_inf:col_sup] = reshaped_im
    plt.imshow(reshaped_tensor, cmap='gray')
    
    # Remove ugly plot ticks
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.label1.set_visible(False)
    
    return fig

#################################################
# Show the latent space
#################################################

def plot_latent_space(model, num_real, classes):
    """
    Plot the latent space of a model 
    Args:
        model: torch.nn.Module 
            A VAE or GAN.
        num_real: int, optional
            Number of real datapoints to show on the latent space of the 
            model as reference. Defaults to 200.
        classes: list of ints
            class labels used to train the model.
        """
    fig = plt.figure(figsize=(5, 4))
    
    # n_class samples are shown per digit, each with a specific color and marker
    n_class = int(num_real/len(classes))
    colors = sns.color_palette("light:salmon", n_colors=len(classes))
    markers = ["o","x","*","x","o","p","s","*","p","s"]
    
    for cl_idx, cl in enumerate(classes):
        # Sample data for this class
        loader=dataset_utils.create_loader(n_class, n_class, [cl])
        for imgs,_ in loader:
            # Encode images
            z=model.encode(imgs.reshape(-1,784).to(model.device))
            if get_name(model) == "VAE":
                z = z[0] # encoder mean only
            z=z.detach().cpu()
        
        plt.scatter(
            z[:,0],
            z[:,1],
            color=colors[cl_idx],
            label=cl,
            marker=markers[cl_idx],
            linewidths=0.1
            )
    plt.legend(title="Digit")
    
    return fig

def jac_plot(model, condition=True, num_real=500, res=10, 
            xmin=-1,xmax=1,ymin=-1,ymax=1):
    """
    Plot either 1) the log condition number of the jacobian; or 2) the log
    magnitude of the metric tensor (Jacobian^T . Jacobian). Here, we reference
    the jacobian of the generator of a model. The plots are done along a 
    grid of points on its latent space.
    
    Args:
        model: torch.nn.Module 
            A VAE or GAN.
        condition: boolean, optional
            whether to plot the condition number of the jacobian or the
            magnitude of the metric tensor.
        num_real: int, optional
            Number of real datapoints to show on the latent space of the 
            model as reference. Defaults to 500.
        res: int, optional
            Resolution of each dimension of the grid of the latent space. 
            Defaults to 10.
        xmin (int, optional): Grid limit. Defaults to -1.
        xmax (int, optional): Grid limit. Defaults to 1.
        ymin (int, optional): Grid limit. Defaults to -1.
        ymax (int, optional): Grid limit. Defaults to 1.
    """
    fig = plt.figure(figsize=(5,4))
    name = get_name(model)
    
    # Get the generator
    if get_name(model) == "GAN":
        generator=model.generator
    else:
        generator=model.decode
    
    # Add some real points to the plot.
    if name != "GAN":
        fig = plot_latent_space(model, num_real, list(range(10)))
    
    # Make a grid of the latent space.
    z1,z2=torch.linspace(xmin, xmax, res), torch.linspace(ymin, ymax, res)
    z = torch.cartesian_prod(z1, z2).to(model.device) 
    
    # get the jacobian of the generator at each point
    m = np.zeros(len(z))
    for i in range(len(z)):
        # Compute the jacobian of the generator
        J = jacobian(generator, z[i].view(1,-1))
        J = J.view(784,-1).detach().cpu()
        if condition:
            m[i] = np.log(np.linalg.cond(J) + 1e-10) # log condition number
        else:
            m[i] = np.log(np.transpose(J).mm(J).det() + 1e-10) # log determinant
    
    # Visualize
    z=z.detach().cpu()
    plt.scatter(z[:, 0], z[:, 1], c=m, zorder=-1) 
    plt.title(name + " Generator's Jacobian")
    cbar = plt.colorbar()
    if condition: cbar.set_label("Log-condition number of the jacobian")
    else: cbar.set_label("Log-determinant of the metric tensor")
    
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    
    return fig


######################################
# Plot interpolations
######################################
def linear_interpolation(model,x0,x1,ts=10,num_real=500):
    """
    Plot various generated images via linear interpolation of the latent
    representations of two data points. 
    
    Args:
        model: torch.nn.Module 
            A VAE or GAN.
        x0: torch.tensor
            One image for the interpolation
        x1: torch.tensor
            Another image for the interpolation
        ts (int, optional): 
            Number of equidistant points to sample on the line segment
            between the latent representations of x0 and x1. Defaults to 10.
        num_real: int, optional
            Number of real datapoints to show on the latent space of the 
            model as reference. Defaults to 500.
    """
    name = get_name(model) 
    x0 = x0.to(model.device).view(-1,784)
    x1 = x1.to(model.device).view(-1,784)
    
    # Produce the latent representation of interpolation limits
    z0, z1 = model.encode(x0), model.encode(x1)
    if name == "VAE":z0, z1 = z0[0], z1[0]
    z0, z1 = z0.detach().cpu(), z1.detach().cpu()
    
    fig,axs = plt.subplots(1,ts,figsize=(18,6))
    
    z = torch.zeros(ts,2)
    ts = np.linspace(0,1,ts)
    
    for i,t in enumerate(ts):
        z[i,:] = (z0*t+(1-t)*z1) # linear interpolation
        
        # Data generation
        img = model.decode(z[i,:].to(model.device))
        img = img.detach().cpu().view(28,28)
        
        # plotting
        axs[i].imshow(img,cmap="gray")
        t = np.round(t,2)
        axs[i].title.set_text('t='+"{:.2f}".format(t))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    # Plot the latent space
    latent = plot_latent_space(model, num_real, list(range(10)))
    
    latent.scatter(z[:,0], z[:,1], c=ts, marker='*', s=100, cmap="cool")
    cbar = latent.colorbar()
    cbar.ax.tick_params(size=0)
    cbar.set_ticks([0,1])
    cbar.set_ticklabels(["Start", "End"])
    
    return latent,fig

def get_name(obj):
    """
    Gets the name of the class of an object
    """
    return type(obj).__name__