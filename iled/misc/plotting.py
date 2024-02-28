import matplotlib.pyplot as plt
import numpy as np


def plot_losses(stats, ax=None):
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    tr_mean_loss = np.array(stats.losses)
    tr_loss_std = np.array(stats.losses_std)
    x = np.arange(len(tr_mean_loss))
    va_mean_loss = np.array(stats.val_losses)
    va_loss_std = np.array(stats.val_losses_std)
    nTmax_lst = np.array(stats.nTmax_lst)

    x_val = np.linspace(0, len(tr_mean_loss) - 1, len(va_mean_loss))

    lns1 = plt.semilogy(tr_mean_loss, label="Train Loss")
    plt.fill_between(
        x,
        tr_mean_loss - tr_loss_std,
        tr_mean_loss + tr_loss_std,
        alpha=0.3,
        color="tab:blue",
    )
    lns2 = plt.semilogy(x_val, va_mean_loss, label="Validation Loss")
    plt.fill_between(
        x_val,
        va_mean_loss - va_loss_std,
        va_mean_loss + va_loss_std,
        alpha=0.3,
        color="tab:orange",
    )

    plt.ylim(tr_mean_loss.min() * 0.8, 1.1 * tr_mean_loss[0])
    new_ax = ax.twinx()
    x_metrics = np.linspace(0, len(tr_mean_loss) - 1, len(nTmax_lst))
    lns3 = plt.plot(
        x_metrics, nTmax_lst, color="tab:green", zorder=3, label="Trajectory Length"
    )

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper center")
    plt.title("Losses")
    ax.set_xlabel("Epoch")
