import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append("../../")
import iled
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def simulate_traj(traj, trainer, Tf):

    batch = torch.stack([iled.qol.to_tensor(traj)])
    batch = trainer.scaler.scaling(batch)
    ret = trainer.model(batch, NTmax=Tf)
    truth = iled.qol.to_numpy(traj)
    reconstruction = ret["reconstruction"][0]
    rec_forecast = ret["reconstructed_forecast"][0]
    true_latents = iled.qol.to_numpy(ret["true_latents"][0])
    latent_forecast = iled.qol.to_numpy(ret["latent_forecast"][0])
    if ret['dynamics_parts'][0] is None:
        linear_part, nl_part = None,None
    else:
        linear_part = iled.qol.to_numpy(ret['dynamics_parts'][0][0])
        nl_part = iled.qol.to_numpy(ret['dynamics_parts'][1][0])

    if trainer.scaler:
        reconstruction = trainer.scaler.inverse_scaling(reconstruction)
        rec_forecast = trainer.scaler.inverse_scaling(rec_forecast)

    reconstruction = iled.qol.to_numpy(reconstruction)
    rec_forecast = iled.qol.to_numpy(rec_forecast)

    return truth, reconstruction, rec_forecast, true_latents, latent_forecast, linear_part, nl_part


def colorbar_ticks(cbar, field, n_ticks=4, form="float"):
    m0 = field.min()  # colorbar min value
    m1 = field.max()  # colorbar max value
    ticks = np.linspace(0, 1, n_ticks)
    labels = np.linspace(m0, m1, n_ticks)
    cbar.set_ticks(labels)
    if form == "float":
        cbar.set_ticklabels(["{:.2f}".format(m) for m in labels])
    else:
        cbar.set_ticklabels(["{:.0e}".format(m) for m in labels])


def prediction_plot(traj, trainer, Tvis=None, Tf=None, autoencoder=False, save_path = None, name='default'):

    if Tvis is None:
        Tvis = trainer.config.target_length
    if Tf is None:
        Tf = Tvis

    n_warmup = trainer.model.get_n_warmup()

    if autoencoder:
        truth, reconstruction, rec_forecast, true_latents, latent_forecast, linear_part, nl_part = simulate_traj(
                traj, trainer, n_warmup+2
            )
    else:
        truth, reconstruction, rec_forecast, true_latents, latent_forecast, linear_part, nl_part = simulate_traj(
        traj, trainer, Tf
        )

    iv = 1

    field = truth[:Tvis, iv, :].T
    if autoencoder:
        field_pred = reconstruction[:Tvis, iv, :].T
    else:
        field_pred = rec_forecast[:Tvis, iv, :].T

    matplotlib.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = gridspec.GridSpec(3, 11, hspace=0.25, wspace=0)

    

    w_1 = 8
    pad = 0.01

    x = np.arange(Tvis)
    y = np.linspace(0, 20, traj.shape[-1])

    ax1 = plt.subplot(gs[0, :w_1])
    ola = plt.contourf(x, y, field, 100, cmap=plt.cm.seismic)
    cbar = plt.colorbar(ola, pad=pad)
    colorbar_ticks(cbar, field)
    plt.title(r"$v(x,t)$")
    plt.ylabel(r"$x$")
    plt.xticks([])

    ax2 = plt.subplot(gs[1, :w_1])
    plt.sca(ax2)
    ola = plt.contourf(x, y, field_pred, 100, cmap=plt.cm.seismic)
    cbar = plt.colorbar(ola, pad=pad)
    colorbar_ticks(cbar, field_pred)
    plt.title(r"$\hat{v}(x,t)$")
    plt.ylabel(r"$x$")
    plt.xticks([])

    ax3 = plt.subplot(gs[2, :w_1])
    plt.sca(ax3)
    error = abs(field - field_pred)
    ola = plt.contourf(x, y, error, 100, cmap=plt.cm.Reds)
    cbar = plt.colorbar(ola, pad=pad)
    colorbar_ticks(cbar, error, form="exp")
    plt.title(r"$\vert v(x,t)- \hat{v}(x,t) \vert$")
    plt.ylabel(r"$x$")
    plt.xlabel(r"$t (s)$")

    ax4 = plt.subplot(gs[:, w_1:])
    plt.sca(ax4)
    plt.title(r"Latent space trajectory $T_f$ = {:}".format(Tf))
    plt.scatter(true_latents[:, 0], true_latents[:, 1], s=6, label="Truth")
    if not autoencoder:
        plt.scatter(latent_forecast[:, 0], latent_forecast[:, 1], s=4, label="Predicted")
    plt.legend()
    plt.xlabel(r"$z_1$")
    plt.xticks([])
    plt.ylabel(r"$z_2$")
    plt.yticks([])

    if save_path:
        fig.savefig(osp.join(save_path,name,'_trajectory.png'),dpi=300,bbox_inches='tight')
    else:
        plt.show()


    if linear_part is None: return

    linear_norm = np.linalg.norm(linear_part,axis=-1)
    nl_norm = np.linalg.norm(nl_part,axis=-1)

    print(linear_norm.mean(),nl_norm.mean())

    fig2 = plt.figure(figsize=(16, 2))
    plt.semilogy(linear_norm,lw=3,label='linear part')
    plt.semilogy(nl_norm,lw=3,label='Non-linear part')
    plt.legend()
    plt.show()


def autoencoder_plot(traj,trainer, Tvis=None, Tf=None):
    prediction_plot(traj,trainer,Tvis=Tvis,Tf=Tf,autoencoder=True)


def encoded_plot(traj, trainer, Tvis=None, Tf=None, save_path = None):

    if Tvis is None:
        Tvis = trainer.config.target_length
    if Tf is None:
        Tf = Tvis

    n_warmup = trainer.model.get_n_warmup()

    
    truth, reconstruction, _, true_latents, _, _ , _ = simulate_traj(
            traj, trainer, n_warmup+2
        )

    field = truth[:Tvis, :, :].T

    matplotlib.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(16, 4), constrained_layout=True)
    gs = gridspec.GridSpec(2, 11, hspace=0.25, wspace=0.5)

    w_1 = 8
    pad = 0.01

    x = np.arange(Tvis)
    y = np.linspace(0, 20, traj.shape[-1])

    ax1 = plt.subplot(gs[0, :w_1])
    ola = plt.contourf(x, y, field[:,0], 100, cmap=plt.cm.seismic)
    plt.xticks([])
    plt.title(r"$u(x,t)$")
    plt.ylabel(r"$x$")

    ax2 = plt.subplot(gs[1, :w_1])
    plt.sca(ax2)
    ola = plt.contourf(x, y, field[:,1], 100, cmap=plt.cm.seismic)
    plt.title(r"$v(x,t)$")
    plt.ylabel(r"$x$")
    plt.xlabel(r'$t(s)$')

    ax4 = plt.subplot(gs[:, w_1:])
    plt.sca(ax4)
    plt.title(r"Latent space trajectory")
    plt.scatter(true_latents[:, 0], true_latents[:, 1], s=6, label="Truth")
    plt.xlabel(r"$z_1$")
    #plt.xticks([])
    plt.ylabel(r"$z_2$")
    #plt.yticks([])
