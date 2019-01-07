
#===========================================================
#Copyright(c)2018, EMBL-EBI
#All rights reserved.
#NAME:		scanpy_utils.py
#ABSTRACT:	A scRNA-Seq workflow
#DATE:		Mon Dec 25 14:58:31 2017
#Usage:		
#VERSION: 	0.01
#AUTHOR: 	Miao Zhichao
#CONTACT: 	chichaumiau AT gmail DOT com
#NOTICE: This is free software and the source code is freely
#available. You are free to redistribute or modify under the
#conditions that (1) this notice is not removed or modified
#in any way and (2) any modified versions of the program are
#also available for free.
#		** Absolutely no Warranty **
#===========================================================

# hide codes
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="code"></form>''')

# import scanpy
import pandas as pd
import numpy as np
import scanpy.api as sc
from io import StringIO  # got moved to io in python3.
import requests

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80)  # low dpi (dots per inch) yields small inline figures
sc.logging.print_version_and_date()
# we will soon provide an update with more recent dependencies
sc.logging.print_versions_dependencies_numerics()

# clean obs (metadata) before saving the file
def sc_clean_obs(adata):
    adata.obs.drop(columns=[col for col in adata.obs.columns if len(adata.obs[col][~adata.obs[col].isna()].unique())<2],inplace=True)

# bGPLVM
import GPy
def calc_bgplvm(X, dim=2, max_iters=200):
    bgplvm = GPy.models.BayesianGPLVM(X,
          input_dim=dim,
          kernel=GPy.kern.RBF(dim, ARD=True))
    bgplvm.optimize(messages=True, max_iters=max_iters)
    return np.asarray(bgplvm.X.mean[:, :])

# optimize the regress out function
import numpy as np
import scipy as sp
import warnings
import patsy
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from pandas.api.types import is_categorical_dtype
from anndata import AnnData
from scanpy import settings as sett
from scanpy import logging as logg

def sc_pp_regress_out(adata, keys, n_jobs=None, copy=False):
    """Regress out unwanted sources of variation.
    Uses simple linear regression. This is inspired by Seurat's `regressOut`
    function in R [Satija15].
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    keys : str or list of strings
        Keys for observation annotation on which to regress on.
    n_jobs : int
        Number of jobs for parallel computation.
    copy : bool (default: False)
        If an AnnData is passed, determines whether a copy is returned.
    Returns
    -------
    Depening on `copy` returns or updates `adata` with the corrected data matrix.
    """
    logg.info('regressing out', keys, r=True)
    if issparse(adata.X):
        logg.info('... sparse input is densified and may '
                  'lead to huge memory consumption')
    adata = adata.copy() if copy else adata
    if isinstance(keys, str): keys = [keys]
    if issparse(adata.X):
        adata.X = adata.X.toarray()
    if n_jobs is not None:
        logg.warn('Parallelization is currently broke, will be restored soon. Running on 1 core.')
    n_jobs = sett.n_jobs if n_jobs is None else n_jobs

    cat_keys = []
    var_keys = []
    for key in keys:
        if key in adata.obs_keys():
            if is_categorical_dtype(adata.obs[key]):
                cat_keys.append(key)
            else:
                var_keys.append(key)
    cat_regressors = None
    if len(cat_keys)>0:
        cat_regressors = patsy.dmatrix("+".join(cat_keys), adata.obs)
    var_regressors = None
    if len(var_keys)>0:
        var_regressors = np.array(
            [adata.obs[key].values if key in var_keys
             else adata[:, key].X for key in var_keys]).T
    if cat_regressors is None:
        regressors = var_regressors
        if regressors is None:
            logg.warn('No correct key provided. Data not regressed out.')
            return adata
    else:
        if var_regressors is None:
            regressors = cat_regressors
        else:
            regressors = np.hstack((cat_regressors,var_regressors))

    regressors = np.c_[np.ones(adata.X.shape[0]), regressors]
    len_chunk = np.ceil(min(1000, adata.X.shape[1]) / n_jobs).astype(int)
    n_chunks = np.ceil(adata.X.shape[1] / len_chunk).astype(int)
    chunks = [np.arange(start, min(start + len_chunk, adata.X.shape[1]))
              for start in range(0, n_chunks * len_chunk, len_chunk)]

    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    def _regress_out(col_index, responses, regressors):
        try:
            if regressors.shape[1] - 1 == responses.shape[1]:
                regressors_view = np.c_[regressors[:, 0], regressors[:, col_index + 1]]
            else:
                regressors_view = regressors
            result = sm.GLM(responses[:, col_index],
                            regressors_view, family=sm.families.Gaussian()).fit()
            new_column = result.resid_response
        except PerfectSeparationError:  # this emulates R's behavior
            logg.warn('Encountered PerfectSeparationError, setting to 0 as in R.')
            new_column = np.zeros(responses.shape[0])
        return new_column

    def _regress_out_chunk(chunk, responses, regressors):
        chunk_array = np.zeros((responses.shape[0], chunk.size),
                               dtype=responses.dtype)
        for i, col_index in enumerate(chunk):
            chunk_array[:, i] = _regress_out(col_index, responses, regressors)
        return chunk_array

    for chunk in chunks:
        # why did this break after migrating to dataframes?
        # result_lst = Parallel(n_jobs=n_jobs)(
        #     delayed(_regress_out)(
        #         col_index, adata.X, regressors) for col_index in chunk)
        result_lst = [_regress_out(
            col_index, adata.X, regressors) for col_index in chunk]
        for i_column, column in enumerate(chunk):
            adata.X[:, column] = result_lst[i_column]
    logg.info('finished', t=True)
    logg.hint('after `sc.pp.regress_out`, consider rescaling the adata using `sc.pp.scale`')
    return adata if copy else None

# optimize violin plot
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from matplotlib import pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import is_color_like
import seaborn as sns

from scanpy import settings
from scanpy.plotting import utils
from scanpy.plotting.utils import scatter_base, scatter_group
from scanpy.utils import sanitize_anndata

#'axs = sc_pl_violin(adata, 'n_genes', group_by= "name",
                   #'jitter=0.4, multi_panel=True, col_wrap = 6, log =True)

def sc_pl_violin(adata, keys, group_by=None, log=False, use_raw=True, jitter=True,
           size=1, scale='width', order=None, multi_panel=False, show=None,
           save=None, col_wrap = 6, ax=None, **kwargs):
    """Violin plot [Waskom16]_.
    Wraps `seaborn.violinplot` for :class:`~scanpy.api.AnnData`.
    Parameters
    ----------
    adata : :class:`~scanpy.api.AnnData`
        Annotated data matrix.
    keys : `str` or list of `str`
        Keys for accessing variables of `.var_names` or fields of `.obs`.
    group_by : `str` or `None`, optional (default: `None`)
        The key of the observation grouping to consider.
    log : `bool`, optional (default: `False`)
        Plot on logarithmic axis.
    use_raw : `bool`, optional (default: `True`)
        Use `raw` attribute of `adata` if present.
    multi_panel : bool, optional
        Show fields in multiple panels. Returns a `seaborn.FacetGrid` in that case.
    jitter : `float` or `bool`, optional (default: `True`)
        See `seaborn.stripplot`.
    size : int, optional (default: 1)
        Size of the jitter points.
    order : list of str, optional (default: `True`)
        Order in which to show the categories.
    scale : {'area', 'count', 'width'}, optional (default: 'width')
        The method used to scale the width of each violin. If 'area', each
        violin will have the same area. If 'count', the width of the violins
        will be scaled by the number of observations in that bin. If 'width',
        each violin will have the same width.
    show : bool, optional (default: `None`)
         Show the plot.
    save : `bool` or `str`, optional (default: `None`)
        If `True` or a `str`, save the figure. A string is appended to the
        default filename. Infer the filetype if ending on \{'.pdf', '.png', '.svg'\}.
    ax : `matplotlib.Axes`
         A `matplotlib.Axes` object.
    **kwargs : keyword arguments
        Are passed to `seaborn.violinplot`.
    Returns
    -------
    A `matplotlib.Axes` object if `ax` is `None` else `None`.
    """
    sanitize_anndata(adata)
    if group_by is not None and isinstance(keys, list):
        raise ValueError('Pass a single key as string if using `group_by`.')
    if isinstance(keys, str): keys = [keys]
    obs_keys = False
    for key in keys:
        if key in adata.obs_keys(): obs_keys = True
        if obs_keys and key not in set(adata.obs_keys()):
            raise ValueError(
                'Either use observation keys or variable names, but do not mix. '
                'Did not find {} in adata.obs_keys().'.format(key))
    if obs_keys:
        obs_df = adata.obs
    else:
        if group_by is None: obs_df = pd.DataFrame()
        else: obs_df = adata.obs.copy()
        for key in keys:
            if adata.raw is not None and use_raw:
                X_col = adata.raw[:, key].X
            else:
                X_col = adata[:, key].X
            obs_df[key] = X_col
    if group_by is None:
        obs_tidy = pd.melt(obs_df, value_vars=keys)
        x = 'variable'
        y = 'value'
    else:
        obs_tidy = obs_df
        x = group_by
        y = keys[0]
    if multi_panel:
        sns.set_style('whitegrid')
        g = sns.FacetGrid(obs_tidy, col=x, sharey=False, col_wrap=col_wrap)
        g = g.map(sns.violinplot, y, inner=None, orient='vertical', scale=scale, **kwargs)
        g = g.map(sns.stripplot, y, orient='vertical', jitter=jitter, size=size,
                     color='black').set_titles(
                         col_template='{col_name}').set_xlabels('')
        if log: g.set(yscale='log')
        ax = g
    else:
        ax = sns.violinplot(x=x, y=y, data=obs_tidy, inner=None, order=order,
                            orient='vertical', scale=scale, ax=ax, **kwargs)
        ax = sns.stripplot(x=x, y=y, data=obs_tidy, order=order,
                           jitter=jitter, color='black', size=size, ax=ax)
        ax.set_xlabel('' if group_by is None else group_by.replace('_', ' '))
        if log: ax.set_yscale('log')
    utils.savefig_or_show('violin', show=show, save=save)
    if show == False: return ax

# add log scale to sc.pl.scatter
def sc_pl_scatter(*args, xlog=False, ylog=False, **kwargs):
    fig,ax = plt.subplots()
    sc.pl.scatter(*args, **kwargs, ax=ax, show=False, save=False)
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    plt.show()

def make_unique(dup_list):
	from collections import Counter
	counter = Counter()
	deduped = []
	for name in dup_list:
		new = name + "_%s"%str(counter[name]) if counter[name] else name
		counter.update({name: 1})
		deduped.append(new)
	return(deduped)

# read a group of data and combine
def sc_read_data(id,dff):
    dfs = []
    for name in dff:
        datapath = "../raw_data/%s"%name
        print(name)
        df = sc.read(datapath + '/matrix.mtx', cache=True).transpose()
        df.obs_names = ["%s_%d"%(name,i) for i in range(df.X.shape[0])]
        df.obs["name"] = name
        df.obs["barcodes"] = np.genfromtxt(datapath + '/barcodes.tsv', dtype=str)
        print("#cell before QC: %d"%df.shape[0])
        sc.pp.filter_cells(df, min_genes=200)
        print("#cell after QC: %d"%df.shape[0])
        dfs.append(df)
    adata = dfs[0].concatenate(tuple(dfs[1:]))
    adata.var_names = np.genfromtxt(datapath + '/genes.tsv', dtype=str)[:, 1]
    adata.var_names_make_unique()
    adata.write("../write/%s_df.h5"%id)

def lda(data, n_comps=50, zero_center=True, svd_solver='svd', random_state=0,
        recompute=True, mute=False, return_info=None, copy=False,
        dtype='float32'):
    """Linear Discriminant Analysis.
    Parameters
    ----------
    data : :class:`~scanpy.api.AnnData`, array-like
        Data matrix of shape `n_obs` × `n_vars`.
    n_comps : `int`, optional (default: 50)
        Number of principal components to compute.
    zero_center : `bool` or `None`, optional (default: `True`)
        If True, compute standard PCA from Covariance matrix. If False, omit
        zero-centering variables, which allows to handle sparse input
        efficiently. If None, defaults to True for dense and to False for sparse
        input.
    svd_solver : `str`, optional (default: 'auto')
        SVD solver to use. Either 'arpack' for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or 'randomized' for the randomized algorithm
        due to Halko (2009). "auto" chooses automatically depending on the size
        of the problem.
    random_state : `int`, optional (default: 0)
        Change to use different intial states for the optimization.
    recompute : `bool`, optional (default: `True`)
        Recompute PC coordinates. If False, use the result of previous calculation.
    return_info : `bool` or `None`, optional (default: `None`)
        If providing an array, this defaults to False, if providing an `AnnData`,
        defaults to `True`.
    copy : `bool` (default: `False`)
        If an `AnnData` is passed, determines whether a copy is returned.
    dtype : str (default: 'float32')
        Numpy data type string to which to convert the result.
    Returns
    -------
    If `data` is array-like and `return_info == True`, only returns `X_pca`,
    otherwise returns or adds to `adata`:
    X_pca : `.obsm`
         PCA representation of data.
    PCs : `.varm`
         The principal components containing the loadings.
    pca_variance_ratio : `.uns`
         Ratio of explained variance.
    pca_variance : `.uns`
         Explained variance, equivalent to the eigenvalues of the covariance matrix.
    """
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        from scanpy import settings as sett  # why is this necessary?
        if ('X_lda' in adata.obs
            and adata.obsm['X_lda'].shape[1] >= n_comps
            and not recompute
            and (sett.recompute == 'none' or sett.recompute == 'pp')):
            logg.msg('    not recomputing LDA, using "X_lda" contained '
                     'in `adata.obs` (set `recompute=True` to avoid this)', v=4)
            return adata
        else:
            logg.msg('compute LDA with n_comps =', n_comps, r=True, v=4)
            result = lda(adata.X, n_comps=n_comps, zero_center=zero_center,
                         svd_solver=svd_solver, random_state=random_state,
                         recompute=recompute, mute=mute, return_info=True)
            X_lda, components, lda_variance_ratio, lda_variance = result
            adata.obsm['X_lda'] = X_lda
            adata.varm['PCs'] = components.T
            adata.uns['lda_variance'] = lda_variance
            adata.uns['lda_variance_ratio'] = lda_variance_ratio
            logg.msg('    finished', t=True, end=' ', v=4)
            logg.msg('and added\n'
                     '    \'X_lda\', the PCA coordinates (adata.obs)\n'
                     '    \'PC1\', \'PC2\', ..., the loadings (adata.var)\n'
                     '    \'lda_variance\', the variance / eigenvalues (adata.uns)\n'
                     '    \'lda_variance_ratio\', the variance ratio (adata.uns)', v=4)
        return adata if copy else None
    X = data  # proceed with data matrix
    from scanpy import settings as sett
    if X.shape[1] < n_comps:
        n_comps = X.shape[1] - 1
        logg.msg('reducing number of computed PCs to',
               n_comps, 'as dim of data is only', X.shape[1], v=4)
    zero_center = zero_center if zero_center is not None else False if issparse(X) else True
    #'from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    verbosity_level = np.inf if mute else 0
    if zero_center:
        if issparse(X):
            logg.msg('    as `zero_center=True`, '
                   'sparse input is densified and may '
                   'lead to huge memory consumption', v=4)
            X = X.toarray()
    lda_ = LinearDiscriminantAnalysis(n_components=n_comps, svd_solver=svd_solver)
    X_pca = pca_.fit_transform(X)
    if X_pca.dtype.descr != np.dtype(dtype).descr: X_pca = X_pca.astype(dtype)
    if False if return_info is None else return_info:
        return X_pca, pca_.components_, pca_.explained_variance_ratio_, pca_.explained_variance_
    else:
        return X_pca

import statsmodels.api as sm
#'import patsy
def Brennecke(adata, spike = False):
	if spike:
		spike_genes = adata.var['ENS'].str.startswith("ERCC")
		ens_genes = adata.var['ENS'].str.startswith("ENS")
		cts_spikes = adata[:,spike_genes].X.T
		cts_genes = adata[:,ens_genes].X.T
	else:
		cts_spikes = adata.X.T
		cts_genes = adata.X.T

	mu_cts_spikes = cts_spikes.mean(1)
	mu_cts_genes = cts_genes.mean(1)
	var_cts_spikes = cts_spikes.var(1)
	var_cts_genes = cts_genes.var(1)
	cv2_cts_spikes = var_cts_spikes/mu_cts_spikes**2
	cv2_cts_genes = var_cts_genes/mu_cts_genes**2
	ncell_spikes = (cts_spikes >0).sum(1)
	ncell_genes = (cts_genes >0).sum(1)
	
	minMeanForFit = pd.Series(mu_cts_spikes[cv2_cts_spikes>.3]).quantile(.8)
	useForFit = (mu_cts_spikes >= minMeanForFit)

	x = patsy.dmatrix('altilde', pd.DataFrame({"altilde":1./mu_cts_spikes[useForFit]}))
	y = cv2_cts_spikes[useForFit]

	model = sm.GLM(y,x,family=sm.families.Gaussian())
	res = model.fit()
	#'res.summary()
	
	residual = cv2_cts_genes - (res.params[0] + res.params[1]/mu_cts_genes)
	return(residual)
	
def removeBatchEffect(metadata, exprs, covariate_formula, design_formula='1', rcond=1e-8):
	# This function is a python implementation of the removeBatchEffect function exactly the same as the function in limma (the R package)
	# But the python version is faster and requires less memory.
	#example: regressed=removeBatchEffect(pheno, exprs, "~ age + cancer", "C(batch)")
	# covariate_formula is the variance to be kept, design_formula is the variance to regress out
	
	design_matrix = patsy.dmatrix(design_formula, metadata)
	# these 3 lines are special
	design_matrix = design_matrix[:,1:]
	rowsum = design_matrix.sum(axis=1) -1
	design_matrix=(design_matrix.T+rowsum).T
	
	covariate_formula += ' -1'
	covariate_matrix = patsy.dmatrix(covariate_formula, metadata)

	design_batch = np.hstack((covariate_matrix,design_matrix))
	coefficients, res, rank, s = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
	
	beta = coefficients[-design_matrix.shape[1]:]
	return(exprs - design_matrix.dot(beta).T)
	
def regress_out(metadata, exprs, covariate_formula, design_formula='1', rcond=-1):
	''' Implementation of limma's removeBatchEffect function
	'''
	# Ensure intercept is not part of covariates
	# covariate_formula is the variance to be kept, design_formula is the variance to regress out

	design_formula += ' -1'
	design_matrix = patsy.dmatrix(design_formula, metadata)

	covariate_matrix = patsy.dmatrix(covariate_formula, metadata)
	
	design_batch = np.hstack((covariate_matrix, design_matrix))
	coefficients, res, rank, s = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
	
	beta = coefficients[covariate_matrix.shape[1]:]
	return(exprs - design_matrix.dot(beta).T)

def DoHeatmap(adata, groupby="louvain", topn=20, levels = None):
    genes = pd.DataFrame(adata.uns['rank_genes_groups']['names']).iloc[:20].T.loc[levels].values.flatten()
    adata.raw.var['idx'] = range(adata.raw.X.shape[1])
    genes_idx = adata.raw.var.loc[genes].idx
    gp = adata.obs.groupby(groupby)
    if levels is None: levels = gp.indices
    cells = np.empty(shape=[0, 0], dtype=int)
    xs = []
    xs2 = []
    x=0
    for level in levels:
        cells = np.append(cells, gp.indices[level])
        xs.append(x+len(gp.indices[level]))
        xs2.append(x+len(gp.indices[level])/2.)
        x+=len(gp.indices[level])
    X = adata.raw.X[cells[:, np.newaxis],genes_idx]
    X = sc.pp.scale(X, max_value=10,copy=True)
    ax = sns.heatmap(X.T, cmap="RdYlBu_r")
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.set_xticks(xs2,minor=True)
    ax.set_xticklabels(levels, minor=True)
    ax.set_yticks(range(len(genes)), minor=True)
    ax.set_yticklabels(genes)
    plt.yticks(rotation=1)

def SubsetData(adata, sele, adata_raw):
	adata = adata[sele,:]
	adata1 = adata_raw[adata_raw.obs_names.isin(adata.obs_names),:]
	for col in adata.obs.columns:
		adata1.obs[col] = adata.obs[col]
	return(adata1)

import plotly
plotly.tools.set_config_file(world_readable=False,
							 sharing='private')
plotly.offline.init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go
from scanpy.plotting.palettes import *
def plotly3d(X, cell_types, unique_types, colors = default_20, name = '1'):
	if len(unique_types) > 20: colors = default_26
	if len(unique_types) > 26: colors = default_64
	x,y,z = X[:,1],X[:,2],X[:,3]
	data=[]
	for i,id in enumerate(unique_types):
		pos = np.where(cell_types==id)
		trace = go.Scatter3d(
			x=x[pos],
			y=y[pos],
			z=z[pos],
			mode='markers',
			name=id,
			marker=dict(
				color=colors[i],
				size=3,
				line=dict(
					color=colors[i],
					width=0.5
				),
				opacity=0.8
			)
		)
		data.append(trace)
	layout = go.Layout(
		margin=dict(
			l=0,
			r=0,
			b=0,
			t=0
		)
	)
	fig = go.Figure(data=data, layout=layout)
	plotly.offline.iplot(fig, filename=name)

def plotly4d(X, c, name = '1', colorscale='Jet'):
	x,y,z = X[:,1],X[:,2],X[:,3]
	trace = go.Scatter3d(
		x=x,
		y=y,
		z=z,
		mode='markers',
		marker=dict(
			size=3,
			color=c,                # set color to an array/list of desired values
			colorscale=colorscale,   # choose a colorscale
			opacity=0.8
		)
	)
	layout = go.Layout(
		margin=dict(
			l=0,
			r=0,
			b=0,
			t=0
		)
	)
	fig = go.Figure(data=[trace], layout=layout)
	plotly.offline.iplot(fig, filename=name)

def run_slalom():
	data = slalom.utils.load_txt(dataFile='../write/Hepatocytes_Fetal_noBCE1_exprs.csv',
                             annoFiles=['../../Ref/h.all.v6.1.symbols.gmt','../../Ref/gene_lists_fscLVM.txt'],
                             annoDBs=['MSigDB','custom'], 
                             niceTerms=[True,False])
	I = data['I'] #if loaded from the hdf file change to I = data['IMSigDB']
	#Y: log expresison values 
	Y = data['Y']
	#terms: ther names of the terms
	terms = data['terms']

	#gene_ids: the ids of the genes in Y
	gene_ids = data['genes']

	#initialize FA instance, here using a Gaussian noise model and fitting 3 dense hidden factors
	FA = slalom.initFA(Y, terms,I, gene_ids=gene_ids, noise='gauss', nHidden=0, minGenes=15)
	
	#model training
	FA.train()

	#print diagnostics
	FA.printDiagnostics()
	
	slalom.utils.plotTerms(FA=FA)

	#get factors; analogous getters are implemented for relevance and weights (see docs)
	plt_terms = ['G2m checkpoint','Th2']
	X = FA.getX(terms=plt_terms)
	
	Ycorr = FA.regressOut(terms = ['Cell.cycle'], Yraw = adata.X)
	
	adata1 = anndata.AnnData(Ycorr, obs=adata.obs)

def run_bgpLVM():
	model = GPy.models.BayesianGPLVM(adata.obsm['X_diffmap'][:,1:3], input_dim=1,\
                                 X=np.array(adata.obs['dpt_pseudotime'])[:,None], init='random')
	model.rbf.lengthscale.constrain_fixed(0.3, warning=False)
	model.rbf.variance.constrain_fixed(25., warning=False)

	model.optimize(messages=True)
	
	Xnew = np.linspace(model.X.mean.min(), model.X.mean.max())[:, None]
	Ynew = model.predict(Xnew)[0]

	figsize(5,5)
	plt.scatter(adata.obsm['X_diffmap'][:,1],adata.obsm['X_diffmap'][:,2], c=adata.obs['dpt_pseudotime'])
	plt.colorbar();
	plt.plot(*Ynew.T, lw=2, c='r');
	plt.grid("off")

	np.save('../write/Hepatocytes_Fetal_best_GPy.npy', model.param_array)

	np.savetxt("../write/Hepatocytes_Fetal_best_GPy.csv", Ynew, delimiter=",")
	return

def fit_gplvm():
	import os
	for i,g in enumerate(adata.var_names):
		print(g)
		os.system("bsub python GPy_GPRegression.py write/Hepatocytes_Fetal_noBCEcc_reg.h5 %s -o log -e error"%g)
	
	Ys = []
	for i,g in enumerate(adata.var_names):
		print(i)
		df = pd.read_csv("GPy/%s.csv"%g,index_col=0, header=None)
		Ys.append(df[1])
	Y = np.array(Ys)
	df100 = pd.DataFrame(Y,index = adata.var_names)
	ds = pd.Series(df100.values.argmax(1),index=df100.index).sort_values(ascending=False)
	from sklearn import preprocessing
	X = preprocessing.scale(df100.loc[ds.index], 1)
	sns.heatmap(X,yticklabels=False,
				xticklabels=False, cmap=cm.RdBu_r)
	plt.xlabel("Pseudotime")
	plt.ylabel("Genes")


#'if __name__ == '__main__':
	#'x=""
