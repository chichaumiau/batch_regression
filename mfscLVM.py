#!/usr/local/bin/python

import fscLVM
import sys
import numpy as np
import pandas as pd
import patsy

import platform
if platform.system() == 'Darwin':
	fanno='/Users/zmiao/Software/f-scLVM-devel/data/xxx.txt'
elif platform.system() == 'Linux':
	fanno='/nfs/research2/teichmann/chichau/Software/f-scLVM/data/fscLVM.gmt.txt'


def run_fscLVM(fin,fphn,fout):
	data = fscLVM.utils.load_txt(dataFile=fin,annoFiles=fanno,annoDBs='MSigDB', dataFile_delimiter=',')
	phn=pd.read_csv(fphn)
	df=pd.read_csv(fin)

	I = data['I']
	Y = data['Y']
	terms = data['terms']
	gene_ids = data['genes']

	X = np.asarray(patsy.dmatrix('C(batch)', phn))
	
	#start
	import time
	t = time.time()
	FA = fscLVM.initFA(Y.astype('float64'), terms,I,noise='gauss', nHidden=1, nHiddenSparse=0, do_preTrain=False, pruneGenes=False, minGenes=15, covariates=X)
	
	FA.train()
	terms=['covariate%d'%i for i in range(X.shape[1])]
	Ycorr=FA.regressOut(terms=terms,use_latent=False, use_lm = False)
	open('time/%s.time'%fout.split('/')[-1],'w').write('%s'%(time.time()-t))

	## Ycorr=FA.regressOut(terms=terms,use_latent=False, use_lm = False, Yraw = Y)
	#end
	
	df1=pd.DataFrame(Ycorr.T,columns=df.index)
	df1=df1.set_index(df.columns.values)
	df1.to_csv(fout)

run_fscLVM(sys.argv[1],sys.argv[2],sys.argv[3])
