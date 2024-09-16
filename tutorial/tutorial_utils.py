from EM import *
from EMk import *
from NMF import ExtendedNMF as NMF


######## get ground truth params ########
def get_gamma_params(plots_path, M, K, cv):
    if (plots_path/f'true_gamma_params.pkl').exists():
        true_gamma_params = Params.load_params(plots_path, f'true_gamma_params')
        a, b, mu, var = true_gamma_params.a, true_gamma_params.b, true_gamma_params.H, true_gamma_params.var
    else:
        print('Creates new ground truth params', flush=True)
        temp = pd.read_csv(root_path/'data/HforSynData.csv', index_col=0)  # was created using NMF on SCLC data with K = 2
        idx = np.random.choice(a=temp.size, size=M*K, replace=False)
        mu = temp.values.flatten()[idx].reshape(M, K)

        mu[mu < 1e-4] = 1e-4
        var = mu**2 * cv**2
        var[var < 1e-4] = 1e-4
        a, b = get_gamma_ab(mu.T, var.T)
        Params(a=a, b=b, var=var, H=mu).dump_params(plots_path, f'true_gamma_params')

    return a, b


def sample_w(N, K, m_lam=2):
    lam = np.random.uniform(m_lam//2, m_lam, N)[:, None]
    w = np.random.dirichlet([1]*K, N)
    return lam * w

######## sample data ########

def sample_data(N, M, K, a, b, w):
    h = gamma.rvs(a=a, scale=1/b, size=(N, K, M))
    r = w[:, :, None] * h
    Y = np.random.poisson(r)
    x = np.sum(Y, axis=1)
    r = r.transpose((0, 2, 1))
    h = h.transpose((0, 2, 1))
    return h, Y, r, x

######## get VarNMF start params ########

def get_em_start(nmf_params):
    mu = nmf_params.H
    mu[mu < EPS] = EPS
    a_em, b_em = get_gamma_ab(mu, np.square(mu))  # cv=1
    w_em = nmf_params.W
    w_em[w_em < EPS] = EPS
    return Params(W=w_em, a=a_em, b=b_em, H=(a_em/b_em))


def get_em_W_start(x, dims, H):
    nmf_obj = NMF(dims, init='random', beta='KL', solver='mu', max_iter=50000)
    start_W, _, _ = nmf_obj.fit_W(x, H=H.T, return_ll=True)
    return start_W


def get_em_test_start(x, train_em_params, dims):
    test_start_params = Params.copy2(train_em_params)
    test_start_params.W = get_em_W_start(x, dims, test_start_params.H)
    return test_start_params


######## normalizing the results ########
def normalize(true_params, est_params, comp_idx=None):
    est_params = Params().copy(est_params)

    if est_params.a is not None:
        est_params.H = est_params.a / est_params.b

    if comp_idx is None:
        corr_mat = np.corrcoef(true_params.H, est_params.H, rowvar=False)[-est_params.H.shape[1]:, :true_params.H.shape[1]]
        comp_idx = linear_sum_assignment(corr_mat.T, maximize=True)[1]
    est_params.W = est_params.W[:, comp_idx]
    est_params.H = est_params.H[:, comp_idx]

    if est_params.a is not None:
        est_params.a = est_params.a[:, comp_idx]
        est_params.b = est_params.b[:, comp_idx]
        est_params.H = est_params.a / est_params.b

    if est_params.posterior_H is not None:
        est_params.posterior_H = est_params.posterior_H[:, :, comp_idx]

    d = np.median(est_params.H, axis=0)
    norm_factor = d / np.median(true_params.H, axis=0)
    est_params.H = est_params.H / norm_factor
    est_params.W = est_params.W * norm_factor

    if est_params.a is not None:
        est_params.b = est_params.b * norm_factor

    if est_params.posterior_H is not None:
        est_params.posterior_H = est_params.posterior_H / norm_factor[None, None, :]
    return est_params


######## plots code ########

def scatter(a, b, log=False, log10=False, diag=True, ma=0, ax=None, c='black', s=1,
            label=None, alpha=1, aspect=False, cmap=None, return_scat_obj=False, args={}):
    # Scatters a vs b of ax (default plt). Can apply:
    #   -   log(x+1) transformation (optional)
    #   -   label (optional)
    #   -   color (default black)
    #   -   size of point (default 1)
    #   -   alpha factor (default 1)
    #   -   aspect ratio equal (optional)
    #   -   diagonal line from 0 to ma (if provided) or to max(a,b) (optional)

    ax_ = ax if ax is not None else plt
    if log:
        a = a+1
        b = b+1
        ax_.semilogx(base=2)
        ax_.semilogy(base=2)
    elif log10:
        a = a+1
        b = b+1
        ax_.semilogx(base=10)
        ax_.semilogy(base=10)
    if diag:
        if ma == 0: ma = np.max([a.max(), b.max()])
        ax_.plot([0, ma], [0, ma], linewidth=1, c='black')
    scat = ax_.scatter(a, b, s=s, c=c, alpha=alpha, label=label, cmap=cmap, **args)
    if aspect:
        if ax is None: ax_.gca().set_aspect('equal')
        else: ax_.set_aspect('equal')
    if return_scat_obj: return ax_, scat
    else: return ax_


def add_labels(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
               xticks=None, xticklabels=None, xtickrotation=None,
               yticks=None, yticklabels=None, ytickrotation=None,
               aspect=False, legend=False, ax=None):
    # Applies labels (if provided)

    if ax is None or ax is plt:
        if title is not None: plt.title(title)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if xticks is not None:
            if xtickrotation is not None: plt.xticks(xticks, xticklabels, rotation=xtickrotation)
            else: plt.xticks(xticks, xticklabels)
        if yticks is not None:
            if ytickrotation is not None: plt.yticks(yticks, yticklabels, rotation=ytickrotation)
            else: plt.yticks(yticks, yticklabels)
        if aspect: plt.gca().set_aspect('equal')
        if legend: plt.legend()
        return plt
    else:
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)
        if xticklabels is not None:
            if xtickrotation is not None: ax.set_xticklabels(xticklabels, rotation=xtickrotation)
            else: ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            if ytickrotation is not None: ax.set_yticklabels(yticklabels, rotation=ytickrotation)
            else: ax.set_yticklabels(yticklabels)
        if aspect: ax.set_aspect('equal')
        if legend: ax.legend()
        return ax

def add_log10_ticks(ax=None, diag=False,
                    xticklabels=np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]),
                    yticklabels=np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000])):
    if ax is None or ax is plt: ax = plt.gca()
    xma, yma = ax.get_xlim()[-1], ax.get_ylim()[-1]


    xticks = xticklabels + 1
    xidx = (np.abs(xma - xticks)).argmin()
    xticks = xticks[:xidx+1]
    xticklabels = xticklabels[:xidx+1]

    yticks = yticklabels + 1
    yidx = (np.abs(yma - yticks)).argmin()
    yticks = yticks[:yidx+1]
    yticklabels = yticklabels[:yidx+1]

    if diag:
        ma = np.max([yticks[-1], xticks[-1]])
        ax.plot([0, ma], [0, ma], linewidth=1, c='black')

    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    add_labels(xticks=xticks, xticklabels=xticklabels,
               yticks=yticks, yticklabels=yticklabels, ax=ax)


def show(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, aspect=False,
         fig=plt, figname=None, tight_layout=True, dpi=300,
         savefig=True, showfig=True, legend=False, savepdf=False):
    # Finishes plot:
    #   -   Applies labels (if provided)
    #   -   Applies legend and tight_layout (optional)
    #   -   Saves to figname (optional)
    #   -   Shows fig (optional - otherwise delete it)

    if type(fig) is plt.Figure and title is not None: fig.suptitle(title)
    if type(fig) is not plt.Figure:
        add_labels(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, aspect=aspect, ax=None)
    else:
        add_labels(xlabel=xlabel, ylabel=ylabel, aspect=aspect, ax=None)

    if legend: plt.legend()
    if tight_layout: fig.tight_layout()
    if savefig and figname is not None: fig.savefig(figname.parent/f'{figname.name}.png', dpi=dpi)
    if savepdf and figname is not None: fig.savefig(figname.parent/f'{figname.name}.pdf')
    if showfig: plt.show()
    else: fig.clf()


def get_fig_grid(K):
    if K % 4 == 0:
        fig_grid = (K // 4, 4)
    elif K % 3 == 0:
        fig_grid = (K // 3, 3)
    elif K % 2 == 0:
        fig_grid = (K // 2, 2)
    else:
        fig_grid = (K // 5, 5)
    return fig_grid


def get_fig_grid_subplots(K, entry_size=(2.5, 3), sharex=False, sharey=False):
    fig_grid = get_fig_grid(K)
    fig, axs = plt.subplots(fig_grid[0], fig_grid[1],
                            figsize=(fig_grid[1] * entry_size[0], fig_grid[0] * entry_size[1]),
                            sharex=sharex, sharey=sharey)
    return fig, axs


def plot_params_scatter(param1, param2, param_name, algo, K, cv, ma=1, log10=False):
    fig, axs = get_fig_grid_subplots(K, sharex=True, sharey=True)
    for k in range(K):
        scatter(param1[..., k], param2[..., k], ax=axs.flat[k], ma=1 if ma == 1 else np.max([param1, param2]), log10=log10)
        add_labels(title=f'comp {k+1}', xlabel='true', ylabel='est.', aspect=1, ax=axs.flat[k])
        if log10: add_log10_ticks(ax=axs.flat[k])
    show(fig=fig, title=f'{param_name} {algo}, RMSE={np.sqrt(np.square(param1 - param2).mean()):.3}, cv={cv}, K={K}')

