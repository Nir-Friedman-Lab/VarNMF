from lib_code.EM_lib import *
from Simulation_code import DataGenerator
from NMF_code.NMF import ExtendedNMF


def create_sig_file():
    sig_path = samples_path/'Signatures'
    p = sig_path.glob('**/*.csv')
    samples_files = [x for x in p if x.is_file()]
    samples = np.sort([x.stem for x in samples_files])
    sig_db = pd.DataFrame(index=samples, columns=np.sort(pd.read_csv(samples_files[0], index_col=0).index)).T
    for f in samples_files:
        print(f.stem)
        sig_db[f.stem] = pd.read_csv(f, index_col=0, usecols=[0, 3])
    sig_db.T.to_csv(samples_path/'sig-normCounts.csv')


def get_filtered_genes(gen_path=root_path):
    # gene_desc = pd.read_csv(root_path/'data'/'GeneDescription.csv', index_col=0)
    # chrXY_genes = pd.read_csv(root_path/'data'/'chrX_chrY_genes2.csv', header=None).values[:, 0]  # not filtered in new list
    # return [g for g in gene_desc[gene_desc.isRefSeq].index if g not in chrXY_genes]
    # return np.loadtxt(gen_path/'data'/'filtered_genes.txt', dtype='object')
    return np.loadtxt(gen_path/'data'/'filtered_genes_new.txt', dtype='object')

def get_house_keeping_genes(gen_path=root_path):

    # gene_desc = pd.read_csv(root_path/'data'/'GeneDescription.csv', index_col=0)
    # chrXY_genes = pd.read_csv(root_path/'data'/'chrX_chrY_genes2.csv', header=None).values[:, 0]
    # filtered_genes = get_filtered_genes(root_path)
    # house_keeping = [g for g in gene_desc[gene_desc.isHouseKeeping].index if g in filtered_genes]
    # np.savetxt(root_path/'data'/'house_keeping_genes.txt', house_keeping, fmt='%s')
    return np.loadtxt(gen_path/'data'/'house_keeping_genes.txt', dtype='object')


def filter_dataset(name='counts.QQnorm', prefix='', samples_path=samples_path, gen_path=root_path):
    data = pd.read_csv(samples_path/f'{prefix}{name}.csv', index_col=0)
    genes = get_filtered_genes(gen_path)
    data.loc[genes].to_csv(samples_path/f'{prefix}{name}-filtered.csv')


def load_datasets(dataset_list=['H', 'H-C', 'H-L-M-N-PH'], counts=False, prefix='', startswith='', background=False,
                  samples_path=samples_path, atlas_path=atlas_path, get_healthy_baseline=True):
    if counts: data = pd.read_csv(samples_path/f"{prefix}counts-filtered.csv", index_col=0)
    else: data = pd.read_csv(samples_path/f"{prefix}counts.QQnorm-filtered.csv", index_col=0)

    if background:
        background = pd.read_csv(samples_path/f"{prefix}background-filtered.csv", index_col=0)

    if atlas_path is not None: atlas = pd.read_csv(atlas_path, index_col=0)
    else: atlas = None

    # gene_scores = np.var(data, axis=1) / (np.mean(data, axis=1) + EPS)
    # q5 = np.quantile(gene_scores, 0.5)
    # data = data[gene_scores > q5]
    # atlas = atlas.loc[data.index.values]

    if 'H' not in dataset_list:
        dataset_list += 'H'

    if dataset_list[0] != 'all':
        if len(startswith) == 0:
            datasets = {label: data[[name for name in data.columns
                                     if not np.all([not name.startswith(l) for l in label.split('-')])]]
                        for label in dataset_list}
        else:
            datasets = {label: data[[name for name in data.columns
                                     if not np.all([not name.startswith(l) for l in label.split('-')])]]
                        for label in [startswith]}
    else:
        datasets = {'all': data,
                    'H': data[[name for name in data.columns if name.startswith('H')]]}

    if get_healthy_baseline: healthy_baseline = np.mean(datasets['H'], axis=1)
    else: healthy_baseline = None

    gene_names = data.index.values
    sample_names = data.columns.values
    return data, atlas, datasets, healthy_baseline, gene_names, sample_names, background


def load_synthetic(gene_names, NMF_inits, K_arr, dataset_list=['H', 'H-C', 'H-L-M'],
                   vars=[0, 3], bg=0, beta='KL', solver='mu', dir_name=None):
    datasets = {}
    for dataset in dataset_list:
        for NMF_init in NMF_inits:
            for var in vars:
                for K in K_arr:
                    dir_name1 = dir_name + '_data/' if dir_name else ''
                    data_path = res_path/f'{dataset}/beta={beta}/solver={solver}/{dir_name1}{NMF_init}_K={K}_var={var}_bg={bg}_'
                    # data_path = res_path + f'{dataset}/beta={beta}/solver={solver}/{dir_name1}{NMF_init}_K={K}_var={var}_bg={bg}_data.csv'
                    dir_name1 = dir_name + '/' if dir_name else ''
                    print(data_path)
                    data = DataGenerator.DataGenerator(data_path).generate().data
                    datasets[f'{dir_name1}{dataset}_{NMF_init}_K={K}_var={var}_bg={bg}'] = pd.DataFrame(data).T.set_index(gene_names)
                    # datasets[f'{dir_name1}{dataset}_{NMF_init}_K={K}_var={var}_bg={bg}'] = pd.read_csv(data_path).T.set_index(gene_names)

    return datasets


def create_df(path_to_samples, sample_file=None):
    if sample_file is not None:
        if sample_file.exists():
            sample_names = np.loadtxt(sample_file, dtype='object')
    else:
        sample_names = [file for file in listdir(path_to_samples) if not file.startswith('.') and file.endswith('.rdata')]
    gene_names = pd.read_csv(path_to_samples/(sample_names[0] + '.counts.csv'), usecols=[0], squeeze=True).values
    for t in ['counts', 'counts.QQnorm', 'background']:
        if t != 'counts.QQnorm':
            df = pd.DataFrame(gene_names, columns=['gene'], dtype='object').set_index('gene')
            for sample in sample_names:
                if sample.endswith('.rdata'):
                    df[sample[:-len('.rdata')]] = pd.read_csv(path_to_samples/(sample + f'.{t}.csv'),
                                                                   names=['gene', sample[:-len('.rdata')]],
                                                                   skiprows=[0]).set_index('gene')
                else:
                    df[sample] = pd.DataFrame([np.nan] * len(df.index), index=df.index)
                    temp = pd.read_csv(path_to_samples/(sample + f'.{t}.csv'),
                                       names=['gene', sample],
                                       skiprows=[0], index_col=0)
                    df[sample].loc[temp.index] = temp.values[:, 0]
        else:
            df = pd.DataFrame(gene_names, columns=['gene'], dtype='object')
            for sample in sample_names:
                if sample.endswith('.rdata'):
                    df[sample[:-len('.rdata')]] = pd.read_csv(path_to_samples/(sample + f'.{t}.csv'),
                                                              names=[sample[:-len('.rdata')]], skiprows=[0],
                                                              squeeze=True).values
                else:
                    df[sample] = pd.DataFrame([np.nan] * len(df.index), index=df.index)
                    temp = pd.read_csv(path_to_samples/(sample + f'.{t}.csv'),
                                                              names=[sample], skiprows=[0],
                                                              squeeze=True, index_col=0)
                    df[sample].loc[temp.index] = temp.values

            df = df.set_index('gene')
        pd.DataFrame.to_csv(df, path_to_samples/f'{t}.csv')

# calc NMF
def calc_NMF(res_path, K_arr, data, prefix='', beta='KL', solver='mu', alpha_W=0., alpha_H=0., l1_ratio=1., init='random',
             const_k_W=None, const_k_H=None, start_k_W=None, start_k_H=None,
             W_all={}, mu_all={}, S=None, col_names=None, ind_names=None, fit_H=True):
    # loss_arr = {}
    for K in K_arr:
        print(K)
        if not (res_path/f'{prefix}W_K={K}.csv').exists():
            NMF_obj = ExtendedNMF(Dims(N=data.shape[0], M=data.shape[1], K=K), init=init, max_iter=10000,
                                  beta=beta, solver=solver, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio,
                                  const_k_W=const_k_W, const_k_H=const_k_H,
                                  start_k_W=start_k_W, start_k_H=start_k_H)

            if fit_H:
                W, mu = NMF_obj.fit(data, S=S,
                                    W=W_all[K].values if K in W_all.keys() else None,
                                    H=mu_all[K].values.T if K in mu_all.keys() else None)
                pd.DataFrame(mu, columns=ind_names).to_csv(res_path / f'{prefix}mu_K={K}.csv')
            else:
                W, mu = NMF_obj.fit_W(data, S=S,
                                    H=mu_all[K].values.T if K in mu_all.keys() else None)

            pd.DataFrame(W, index=col_names).to_csv(res_path/f'{prefix}W_K={K}.csv')
            # loss_arr[K] = NMF_obj.train_loss(data, W, mu)
    # return loss_arr

# get NMF results
def get_NMF_res(K_arr, res_path, sample_names, gene_names, prefix='', return_norm=True):
    W_all, mu_all, W_all, mu_all = {}, {}, {}, {}
    for K in K_arr:
        print(K)
        W_all[K] = pd.read_csv(res_path/f"{prefix}W{'_norm' if return_norm else ''}_K={K}.csv",
                               usecols=np.arange(K+1)[1:]).set_index(sample_names)
        cols = [f'k{i}' for i in W_all[K].columns]
        W_all[K].columns = cols
        temp = pd.read_csv(res_path/f"{prefix}mu{'_norm' if return_norm else ''}_K={K}.csv")
        if temp.shape[0] == K:
            mu_all[K] = temp.T[1:].set_index(gene_names)
        else:
            mu_all[K] = pd.read_csv(res_path/f"{prefix}mu{'_norm' if return_norm else ''}_K={K}.csv", usecols=np.arange(K+1)[1:]).set_index(gene_names)
        mu_all[K].columns = cols
        if K == K_arr[0]:
            idx = np.argsort(W_all[K].sum(axis=0)).index[::-1].values
            W_all[K] = W_all[K][idx]
            W_all[K].columns = cols
            mu_all[K] = mu_all[K][idx]
            mu_all[K].columns = cols
    return W_all, mu_all


# calc_NMF_folds
def gen_folds(data, n_folds):
    fold_size = data.shape[1] // n_folds
    yield data[:, :-fold_size]
    for idx in np.arange(fold_size, data.shape[1], fold_size):
        yield np.concatenate([data[:, :idx], data[:, idx + fold_size:]], axis=1)


def gen_m_folds(data, n_folds):
    fold_size = data.shape[1] // n_folds
    yield data[:, :fold_size]
    for idx in np.arange(fold_size, data.shape[1], fold_size):
        yield data[:, idx:idx + fold_size]


def calc_NMF_folds(data, K_arr, n_folds, prefix=f'H/H'):
    for K in K_arr:
        for i, fold in enumerate(gen_folds(data, n_folds)):
            calc_NMF(res_path, [K], fold.T, prefix=prefix + f'_{i}_')


# get NMF results
def get_NMF_res_folds(n_folds, K_arr, N, M, res_path, prefix=f'H/H', return_norm=True):
    W_all, mu_all = {}, {}
    for K in K_arr:
        W_all[K], mu_all[K] = [], []
        for i in range(n_folds):
            if return_norm:
                temp1, temp2 = get_NMF_res([K], res_path, [str(s) for s in np.arange(N)], [str(s) for s in np.arange(M)],
                                           prefix=prefix + f'_{i}_', return_norm=return_norm)
                W_all[K].append(temp1[K].values)
                mu_all[K].append(temp2[K].values)
    return W_all, mu_all


def sort_mu_W(K_arr, W_all, mu_all, sample_idx, gene_idx, cluster_names):
    K1 = K_arr[0]
    W1 = W_all[K1].loc[sample_idx].copy()
    W_all[K1] = W1
    col_idx1 = cluster_names[:K1]
    mu_all[K1] = mu_all[K1][col_idx1].loc[gene_idx]
    mu_all[K1].columns = cluster_names[:K1]
    for K2 in K_arr[1:]:
        W2 = W_all[K2].loc[sample_idx].copy()
        corr = corr_mat(W1, W2, K1, K2)
        row_ind, col_ind = linear_sum_assignment(corr.T, maximize=True)
        col_idx2 = list(corr.index[col_ind])
        col_idx2 += [c for c in cluster_names[:K2] if c not in col_idx2]

        W2 = W2[col_idx2]
        W2.columns = cluster_names[:K2]

        mu2 = mu_all[K2][col_idx2].loc[gene_idx]
        mu2.columns = cluster_names[:K2]
        W_all[K2] = W2
        mu_all[K2] = mu2

        K1, W1, col_idx1 = K2, W2, col_idx2

    return W_all, mu_all


def W_heatmap(W_norm, K, title=f'W', fig_name=f'W_norm'):
    plt.title(f'{title} K={K}')
    # sns.heatmap(W_norm)
    sns.heatmap(W_norm[:,np.argsort(W_norm.sum(axis=0))])  # sort columns by mass
    # sns.heatmap(W_norm[:,np.argsort(W_norm.sum(axis=0))[::-1]], vmin=0, vmax=1)  # sort columns by mass
    # sns.heatmap(W_norm[:,np.argsort(W_norm.max(axis=0))[::-1]], vmin=0, vmax=1)  # sort column by max
    if SAVE_FIG: plt.savefig(plots_path/f'{fig_name}_K={K}.png')
    if SHOW_FIG: plt.show()
    else: plt.clf()


def W_clustermap(W_norm, K, title=f'W', fig_name=f'W', show=True):
    cg = sns.clustermap(W_norm[:,np.argsort(W_norm.sum(axis=0))], method='complete', metric='correlation', figsize=(5, 10))
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    cg.fig.suptitle(f'{title} K={K}', size=25)
    if SAVE_FIG: plt.savefig(plots_path/f'{fig_name}_K={K}.png')
    if show: plt.show()
    else: plt.clf()
    return cg


def mu_heatmap(mu, K, title=f'mu', fig_name=f'mu'):
    plt.figure(figsize=(5, 10))
    plt.title(f'{title} K={K}')
    sns.heatmap(mu, cmap="YlGnBu")
    if SAVE_FIG: plt.savefig(plots_path/f'{fig_name}_K={K}.png')
    if SHOW_FIG: plt.show()
    else: plt.clf()


def heatmap(d, title, width=300, height=700, colorbar=True,
            xaxis='bottom', yaxis=None, xlabel=None, ylabel=None, ylim=None,
            cmap="RdYlBu_r"):
    plot = d.hvplot.heatmap(
        x='columns',
        y='index',
        title=title,
        cmap=cmap, sort_date=False,
        xaxis=xaxis, yaxis=yaxis,
        xlabel=xlabel, ylabel=ylabel, ylim=ylim,
        rot=70, colorbar=colorbar,
        width=width, height=height).opts(
        toolbar=None,
        fontsize={'title': 10, 'xticks': 5, 'yticks': 4}
    )
    return plot


def W_heatmap_hv(W, K, title=f'W', colorbar=False, yaxis=None, width=30, height=700, mi=0, ma=0,
              xlabel=None, ylabel=None, cmap="RdYlBu_r"):
    if mi != 0:
        temp = pd.DataFrame(columns=W.columns)
        temp.loc['mi'] = [mi] * W.shape[1]
        W = temp.append(W)
    if ma != 0:
        W.loc['ma'] = [ma] * W.shape[1]

    plot = heatmap(W, title=f'{title} K={K}', cmap=cmap,
                   width=width * K + 20 + width if colorbar else 20 + width * K,
                   height=height, colorbar=colorbar, xlabel=xlabel, ylabel=ylabel, yaxis=yaxis)
    return plot


def mu_heatmap_hv(mu, K, title=f'mu', colorbar=False, xaxis='bottom', yaxis=None, width=30, height=1000, mi=0, ma=0):
    if mi != 0:
        temp = pd.DataFrame(columns=mu.columns)
        temp.loc['mi'] = [mi] * mu.shape[1]
        mu = temp.append(mu)
    if ma != 0:
        mu.loc['ma'] = [ma] * mu.shape[1]

    plot = heatmap(mu, title=f'{title} K={K}' if K != 0 else title,
                   width=width * K + 20 + width if colorbar else 20 + width * K,
                   height=height, colorbar=colorbar, xaxis=xaxis, yaxis=yaxis)
    return plot


def data_heatmap_hv(data, K, title=f'data', colorbar=False, xaxis='bottom', yaxis=None, width=500, height=800, mi=0, ma=0, W_labels=None):
    if W_labels:
        temp = pd.DataFrame(columns=data.columns)
        temp.loc['label'] = [mi if l == 0 else ma for l in W_labels]
        data = temp.append(data)

    plot = heatmap(data, title=f'{title} K={K}' if K != 0 else title,
                   width=width, height=height, colorbar=colorbar, xaxis=xaxis, yaxis=yaxis)
    return plot


def corr_heatmap_hv(corr, K1, K2=0, title=f'corr', colorbar=True, yaxis='left', width=400, height=350):
    plot = heatmap(corr, title=f'{title}, K1={K1}' + (f' vs K2={K2}' if K2 > 0 else ''),
                   width=width + 50 if colorbar else width, height=height, colorbar=colorbar,
                   yaxis=yaxis, xlabel=f'K={K1}', ylabel=f'K={K2}' if K2 > 0 else f'K={K1}', ylim=(0,1))
    return plot


def cluster(d, row_names, cluster_names, return_index=False, fixed_col=None, fixed_row=None):
    try:
        clustergrid = sns.clustermap(d, method='complete', metric='correlation',
                                     col_cluster=fixed_col is None, row_cluster=fixed_row is None)
    except:
        plt.clf()
        clustergrid = sns.clustermap(d, method='complete',
                                     col_cluster=fixed_col is None, row_cluster=fixed_row is None)
    plt.clf()
    row_idx = fixed_row if fixed_row is not None else row_names[clustergrid.dendrogram_row.reordered_ind][::-1]
    col_idx = fixed_col if fixed_col is not None else cluster_names[clustergrid.dendrogram_col.reordered_ind]
    d = pd.DataFrame(d[col_idx].loc[row_idx], columns=col_idx).set_index(row_idx)
    if return_index: return d, (row_idx, col_idx)
    else: return d


def cluster_W(W, sample_names, cluster_names, return_index=False, fixed_col=None, fixed_row=None):
    return cluster(W, sample_names, cluster_names, return_index=return_index, fixed_col=fixed_col, fixed_row=fixed_row)


def cluster_mu(mu, gene_names, cluster_names, return_index=False, fixed_col=None, fixed_row=None):
    return cluster(mu, gene_names, cluster_names, return_index=return_index, fixed_col=fixed_col, fixed_row=fixed_row)


def plot_list(maps, cols=0):
    if cols == 0: cols = len(maps)
    subplots = maps[0]
    for i in range(1, len(maps)): subplots += maps[i]
    return subplots.cols(cols)


def plot_corr(a, b, xlabel=None, ylabel=None, logx=True, ax=None, title='corr', plot_diag=True, s=0.03, w_plot=False):
    if title == 'corr': title = f'R={np.round(pearsonr(a, b)[0], 3)}'

    if logx: a_, b_ = np.log(a+1), np.log(b+1)
    else: a_, b_ = a, b

    if not ax: ax = plt.subplot()

    ax.set_title(title)
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')

    ax.scatter(a_, b_, s=s)

    if plot_diag:
        if w_plot:
            ax.plot([0, 1], [0, 1], c='black')
        else:
            ma = np.max([a_, b_])
            mi = np.min([a_, b_])
            ax.plot([mi, ma], [mi, ma], c='black')

    return ax


def corr_mat(a, b, K1, K2):
    mat = pd.DataFrame(np.corrcoef(x=a.values, y=b.values, rowvar=False)[-K2:, :K1],
                       columns=a.columns).set_index(b.columns)
    return mat


def plot_cdf_W(data, ax=plt, label='', llim=1, uselims=True):
    if uselims:
        sorted = [0] + list(np.sort(data)) + [llim]
        p = list(1. * np.arange(len(data)) / (len(data) - 1))
        p = [p[0]] + p + [p[-1]]
    else:
        sorted = list(np.sort(data))
        p = list(1. * np.arange(len(data)) / (len(data) - 1))
    ax.plot(sorted, p, label=label)


def plot_cdf_W_allk(W, sample_idx_dict, K, cluster_names, prefix=''):
    fig, axs = plt.subplots(int(np.ceil(K / 2)), 2, figsize=(8, int(np.ceil(K / 2)) * 3))

    for i, k in enumerate(cluster_names[:K]):
        axs.flat[i].set_title(f'k={i}')
        for l, sample_idx in sample_idx_dict.items():
            ma = np.max(W[k].loc[sample_idx].max())
            plot_cdf_W(W[k].loc[sample_idx], label=l, llim=ma, ax=axs.flat[i])
        axs.flat[i].set_xlabel(f'W')
        axs.flat[i].set_ylabel('p')

    fig.suptitle(f'cdf W, K={K}')
    plt.tight_layout()
    axs.flat[0].legend()
    plt.savefig(plots_path/f'{prefix}W_cdf_K={K}.png')
    plt.clf()


def plot_boxplot_W_allk(W, sample_idx_dict, letters, K, cluster_names, prefix=''):
    fig, axs = plt.subplots(int(np.ceil(K / 2)), 2, figsize=(8, int(np.ceil(K / 2)) * 3))

    for i, k in enumerate(cluster_names[:K]):
        axs.flat[i].set_title(f'k={i}')
        axs.flat[i].boxplot([W[k].loc[sample_idx_dict[l]] for l in letters], notch=True)
        axs.flat[i].set(ylabel='W', axisbelow=True,
                        xticklabels=letters)

    fig.suptitle(f'W, K={K}')
    plt.tight_layout()
    plt.savefig(plots_path/f'{prefix}W_boxplot_K={K}.png')
    plt.clf()


sorting_methods = ['value', 'second', 'zscore', 'diff', 'valuediff']


def z_score(data, axis=1):
    return (data - data.mean(axis=axis).values[:, None]) / (data.std(axis=axis).values[:, None] + EPS)


def get_scores(mu, method='value', l=0.5):
    scores = []
    if method == 'value':
        scores = mu.values / (mu.sum(axis=1).values + EPS)[:, None]
    elif method == 'second':
        scores = np.array([(mu[k].values - mu[[c for c in mu.columns if c != k]].max(axis=1).values) for k in mu.columns]).T
    elif method == 'second2':
        scores = np.array([(mu[k].values - mu[[c for c in mu.columns if c != k]].max(axis=1).values) / (mu[k].values + EPS) for k in mu.columns]).T
    elif method == 'second3':
        scores = np.array([(mu[k].values - mu[[c for c in mu.columns if c != k]].max(axis=1).values) / (mu.sum(axis=1).values + EPS) for k in mu.columns]).T
    elif method == 'zscore':
        scores = z_score(mu).values
    elif method == 'diff':
        scores = np.min([(mu - mu[t].values[:, None]) for t in mu.columns], axis=0)
    elif method == 'valuediff':
        scores = l * mu.values / (mu.sum(axis=1).values + EPS)[:, None] + \
                 (1 - l) * np.min([(mu - mu[t].values[:, None]) for t in mu.columns], axis=0) / (mu.sum(axis=1).values + EPS)[:, None]
    elif method == 'valuemax':
        scores = mu.values / (mu.sum(axis=1).values + EPS)[:, None] * \
                 np.all([(mu >= mu[t].values[:, None]) for t in mu.columns], axis=0)
    return pd.DataFrame(scores, columns=mu.columns).set_index(mu.index)


def get_sorted(mu, method, l=0.5):
    return pd.DataFrame(mu.index.values[np.argsort(get_scores(mu, method, l=l).values, axis=0)[::-1]], columns=mu.columns)


def get_max_genes(sorted_genes, k, n_genes=15):
    return sorted_genes[k][:n_genes]


def save_gene_lists(mu, K, method, l=0.5, n_genes=0, prefix=''):
    sorted_mu = get_sorted(mu, method, l=l)
    if n_genes > 0:
        for k in mu.columns: sorted_mu[k] = sorted_mu[k][:n_genes]
    pd.DataFrame(sorted_mu).to_csv(plots_path/f"{prefix}sorted_genes_{method}_K={K}.csv")


#%%

def calc_rec(lam, W, H, S):
    temp = lam * (W @ H) + S
    return temp

# def grad_w(X, lam, W, H, S):
#     return lam * (X / (calc_rec(lam, W, H, S) + EPS) - 1) @ H.T
#
# def grad_w_tilde(X, lam, W, H, S):
#     return (X / (calc_rec(lam, W, H, S) + EPS) - 1) @ H.T
#
# def grad_h(X, lam, W, H, S):
#     return (lam * W).T @ (X / (calc_rec(lam, W, H, S) + EPS) - 1)


def grad_w_tilde(X, lam, W, H, S, per_j=False):
    temp = X / (calc_rec(lam, W, H, S) + EPS) - 1
    if per_j: return temp.T[:, :, None] * H.T[:, None, :]
    return temp @ H.T


def grad_w(X, lam, W, H, S, per_j=False):
    temp = X / (calc_rec(lam, W, H, S) + EPS) - 1
    if per_j: return lam * temp.T[:, :, None] * H.T[:, None, :]
    return lam * temp @ H.T


def grad_h(X, lam, W, H, S, per_i=False):
    temp = X / (calc_rec(lam, W, H, S) + EPS) - 1
    if per_i: return temp.T[None, :, :] * (lam * W).T[:, None, :]
    return (lam * W).T @ temp


def partial_w_w(X, lam, W, H, S, per_j=False):
    temp = -H.T[None, :, :, None] * H.T[None, :, None, :] * (np.square(lam) * X / np.square(calc_rec(lam, W, H, S) + EPS))[:, :, None, None]
    if per_j: return temp
    else: return temp.sum(axis=1)


def partial_w_w_tilde(X, lam, W, H, S, per_j=False):
    temp = -H.T[None, :, :, None] * H.T[None, :, None, :] * (X / np.square(calc_rec(lam, W, H, S) + EPS))[:, :, None, None]
    if per_j: return temp
    else: return temp.sum(axis=1)


def partial_h_h(X, lam, W, H, S, per_i=False):
    temp = -W[:, None, :, None] * W[:, None, None, :] * (np.square(lam) * X / np.square(calc_rec(lam, W, H, S) + EPS))[:, :, None, None]
    if per_i: return temp
    else: return temp.sum(axis=0)


def partial_h_w(X, lam, W, H, S):
    r = calc_rec(lam, W, H, S)
    temp1 = (lam / (r + EPS))[:, None, None, :]  # (178, 1, 1, 500)
    temp2 = (H[:, :, None, None] * (lam * W)[None, None, :, :]).transpose((2, 0, 3, 1)) * (X / (r + EPS))[:, None, None, :]   # (178, 2, 2, 500)
    temp3 = (X - r)[:, None, None, :] * np.identity(W.shape[1])[None, :, :, None]  # (178, 2, 2, 500)
    # (178, 1, 1, 500), (178, 2, 2, 500), (178, 1, 1, 500) -> (178, 500, 2, 2)
    return (temp1 * (temp3 - temp2)).transpose((0, 3, 1, 2))


def partial_h_w_tilde(X, lam, W, H, S):
    r = calc_rec(lam, W, H, S)
    W_tilde = lam * W
    temp1 = (1 / (r + EPS))[:, :, None, None]
    temp2 = (H.T[None, :, :, None] * W_tilde[:, None, None, :]) * X [:, :, None, None] * temp1
    temp3 = (X - r)[:, :, None, None] * np.identity(W_tilde.shape[1])[None, None, :, :]
    return temp1 * (temp3 - temp2)


def I_w_w_tilde(X, lam, W, H, S, per_j=False):
    # (178, 500) @ (500, 2, 2) -> (178, 2, 2)
    r = calc_rec(lam, W, H, S) + EPS
    temp = 1 / r
    temp[r < EPS ** .25] = 0
    temp = H.T[None, :, :, None] * H.T[None, :, None, :] * temp[:, :, None, None]
    if per_j: return temp
    return temp.sum(axis=1)


def I_h_h(X, lam, W, H, S, per_i=False):
    # (500, 178) @ (178, 2, 2) -> (500, 2, 2)
    r = calc_rec(lam, W, H, S) + EPS
    temp = 1 / r
    temp[r < EPS ** .25] = 0
    W_tilde = lam * W
    temp = W_tilde[:, None, :, None] * W_tilde[:, None, None, :] * temp[:, :, None, None]
    if per_i: return temp
    return temp.sum(axis=0)


def I_h_w_tilde(X, lam, W, H, S):
    r = calc_rec(lam, W, H, S) + EPS
    temp = 1 / r
    temp[r < EPS ** .25] = 0
    W_tilde = lam * W
    temp = H.T[None, :, :, None] * W_tilde[:, None, None, :] * temp[:, :, None, None]
    # temp[temp == 0] = (temp + np.identity(K)[None, None, :, :])[temp == 0]
    return temp


def ll(X, lam, W, mu, S, per_ij=False, per_j=False, per_i=False):
    rec = lam * (W @ mu) + S
    rvs = poisson(rec)
    res = rvs.logpmf(X)
    if per_ij: return res
    if per_j: return res.mean(axis=0)
    if per_i: return res.mean(axis=1)
    else: return res.mean()