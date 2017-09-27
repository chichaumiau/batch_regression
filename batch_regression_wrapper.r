#!/usr/local/bin Rscript

#===========================================================
#Copyright(c)2017, EMBL-EBI
#All rights reserved.
#NAME:		bregress.r
#ABSTRACT:	Batch regression methods
#DATE:		Mon Mar 13 13:40:08 2017
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

get_control_genes<-function(counts,batch){
	suppressPackageStartupMessages(library(edgeR))
	design <- model.matrix(~as.factor(batch))
	y <- DGEList(counts=counts, group=batch)
	y <- calcNormFactors(y, method="TMM")
	y <- estimateGLMCommonDisp(y, design)
	y <- estimateGLMTagwiseDisp(y, design)
	fit <- glmFit(y, design)
	lrt <- glmLRT(fit, coef=2)
	rk<-rank(lrt$table$LR)
	rk <= max(min(rk),400)
}


suppressPackageStartupMessages(library(scater))
library(rmngb)
args = commandArgs(trailingOnly=TRUE)
sce <- readRDS("filtered.rds")
opt<-args[1]
print(dim(sce))

run_limma<-function(x,batch,sample,norm_name){
	suppressPackageStartupMessages(library(limma))
	design <- model.matrix(~as.factor(sample))
	y <- limma::removeBatchEffect(x, batch=as.factor(batch), design=design)
	saveRDS(y, file = sprintf("regressed/%s_limma.rds",norm_name))
}
run_Combat<-function(x,batch,sample,norm_name){
	suppressPackageStartupMessages(library(sva))
	design <- model.matrix(~as.factor(sample))
	y <- sva::ComBat(dat= x[rowVars(x)!=0,],batch=as.factor(batch), mod=design)
	saveRDS(y, file = sprintf("regressed/%s_Combat.rds",norm_name))
}

run_RUVs<-function(x, batch, k, isLog, norm_name){
	suppressPackageStartupMessages(library(RUVSeq))
	genes <- rownames(x)
	differences<-makeGroups(as.factor(batch))
	print(sprintf("RUVs %s k=%d",norm_name,k))
	if(!isLog){
		x<-log(x+1)
	}
	y<-RUVs(x, genes, k = k, differences, isLog =T)
	saveRDS(y$normalizedCounts, file = sprintf("regressed/%s_RUVs_k%d.rds",norm_name,k))
}

run_RUVg_cg<-function(x, batch, sample, k, counts, isLog, norm_name){
	suppressPackageStartupMessages(library(RUVSeq))
	suppressPackageStartupMessages(library(edgeR))

	design <- model.matrix(~as.factor(batch))
	y <- DGEList(counts=counts, group=sample)
	y <- calcNormFactors(y, method="TMM")
	y <- estimateGLMCommonDisp(y, design)
	y <- estimateGLMTagwiseDisp(y, design)
	fit <- glmFit(y, design)

	lrt <- glmLRT(fit, coef=2)
	rk<-rank(lrt$table$LR)
	controls<- (rk <= max(min(rk),400))
	print(sum(controls))
	if(!isLog){
		x<-log(x+1)
	}
	y<-RUVg(x, controls,k = k, isLog =T)
	saveRDS(y$normalizedCounts, file = sprintf("regressed/%s_RUVg_cg_k%d.rds",norm_name,k))
}

run_RUVg<-function(x, k, isLog, norm_name){
	suppressPackageStartupMessages(library(RUVSeq))
	if(!isLog){
		x<-log(x+1)
	}
	y<-RUVg(x, k = k, isLog =T)
	saveRDS(y$normalizedCounts, file = sprintf("regressed/%s_RUVg_k%d.rds",norm_name,k))
}

run_RUVr<-function(x, batch, sample, k, counts, isLog, norm_name){
	suppressPackageStartupMessages(library(RUVSeq))
	suppressPackageStartupMessages(library(edgeR))

	design <- model.matrix(~as.factor(batch))
	y <- DGEList(counts=counts, group=sample)
	y <- calcNormFactors(y, method="TMM")
	y <- estimateGLMCommonDisp(y, design)
	y <- estimateGLMTagwiseDisp(y, design)
	fit <- glmFit(y, design)
	res <- residuals(fit, type="deviance")

	lrt <- glmLRT(fit, coef=2)
	rk<-rank(lrt$table$LR)
	controls<- (rk <= max(min(rk),400))
	print(sum(controls))
	if(!isLog){
		x<-log(x+1)
	}
	y<-RUVr(x, controls, k = k, res, isLog =T)
	saveRDS(y$normalizedCounts, file = sprintf("regressed/%s_RUVr_k%d.rds",norm_name,k))
}

run_PEER<-function(x, sample, k, isLog,norm_name){
	if(!isLog){
		x<-log(x+1)
	}
	library(peer)
	print("alksdjf;ajdskl;")
	model = PEER()
	PEER_setPhenoMean(model,as.matrix(x))
	#design <- model.matrix(~as.factor(sample))
	#PEER_setCovariates(model, as.matrix(design))
	PEER_setCovariates(model, as.matrix(sample))
	print("==============")
	PEER_setNk(model,k)
	PEER_update(model)
	y<-PEER_getResiduals(model)

	colnames(y)<-colnames(x)
	rownames(y)<-rownames(x)
	saveRDS(y, file = sprintf("regressed/%s_PEER_k%d.rds",norm_name,k))
}

run_sva<-function(x, sample, k, norm_name){
	suppressPackageStartupMessages(library(sva))
	suppressPackageStartupMessages(library(limma))
	meta <- data.frame(sample = sample)
	mod = model.matrix(~sample, data=meta)
	mod0 <- model.matrix(~1,data=meta)
	k = num.sv(x,mod,method="leek")
	k = max(2,k)
	svobj <- sva::sva(x,mod,mod0,n.sv=k)
	design <- model.matrix(~as.factor(sample))
	y <- limma::removeBatchEffect(x, batch=svobj$sv, design=design)
	saveRDS(y, file = sprintf("regressed/%s_sva.rds",norm_name))
}

#prepare
norm_name<-args[2]
isLog.dict<-list(Counts=FALSE,logCounts=TRUE,TPM=FALSE,logTPM=TRUE,scater.CPM=FALSE,scater.logCPM=TRUE,edgeR.CPM=FALSE,edgeR.logCPM=TRUE,scater.RLE=FALSE,scater.TMM=FALSE,scran.pool=FALSE,scnorm=FALSE,deseq2.MRN=FALSE, yarn.qsmooth=FALSE )
isLog<-isLog.dict[[norm_name]]
print(norm_name)
x<-readRDS(sprintf("normalized/%s.rds",norm_name))

if(!file.exists("normalized/pData.csv")){
	write.table(pData(sce),"normalized/pData.csv",sep=',')
}
if(!file.exists(sprintf("normalized/%s.csv",norm_name))){
	write.table(t(x),sprintf("normalized/%s.csv",norm_name),sep=',')
}

if(opt=='limma'){
	run_limma(x,droplevels(sce$batch),droplevels(sce$sample),norm_name)
}else if(opt=='Combat'){
	run_Combat(x,droplevels(sce$batch),droplevels(sce$sample),norm_name)
}else if(opt=='RUVg_cg'){
	k=as.integer(args[3])
	run_RUVg_cg(x, droplevels(sce$batch), droplevels(sce$sample), k, counts(sce), isLog, norm_name)
}else if(opt=='RUVg'){
	k=as.integer(args[3])
	run_RUVg(x, k, isLog, norm_name)
}else if(opt=='RUVr'){
	k=as.integer(args[3])
	run_RUVr(x, droplevels(sce$batch), droplevels(sce$sample), k, counts(sce), isLog, norm_name)
}else if(opt=='RUVs'){
	k=as.integer(args[3])
	run_RUVs(x, droplevels(sce$batch), k, isLog, norm_name)
}else if(opt=='fscLVM'){
	cmd=sprintf('python ../lib/mfscLVM.py normalized/%s.csv normalized/pData.csv regressed/%s_fscLVM.csv',norm_name,norm_name)
	print(cmd)
	system(cmd)
}else if(opt=='sva'){ #not used
	run_sva(x, droplevels(sce$sample), norm_name)
}else if(opt=='peer'){
	k=as.integer(args[3])
	run_PEER(x, droplevels(sce$sample), k, isLog, norm_name)
}else if(opt=='scLVM'){ #not used
	return
#~ 		write.table(pData(sce1),sprintf("regressed/pData_%s.csv",sample_name),sep=',')
#~ 		write.table(t(x[!spikes,]),sprintf("regressed/%s.csv",norm_name),sep=',')

#~ 		suppressPackageStartupMessages(library(scLVM))
#~ 		cts<-counts(sce1)[!grepl("^ERCC", featureNames(sce1)), ]
#~ 		ctsERCC<-counts(sce1)[grepl("^ERCC", featureNames(sce1)), ]
#~ 		if(sum(ctsERCC)>0){
#~ 		  techNoise = fitTechnicalNoise(cts,nCountsERCC=ctsERCC, fit_type = 'counts',plot=FALSE)
#~ 		}else{
#~ 		  techNoise = fitTechnicalNoise(cts, fit_type = 'log', use_ERCC = FALSE,plot=FALSE)
#~ 		}
#~ 		write.table(techNoise$techNoiseLog,sprintf("regressed/techNoise_%s.csv",sample_name),sep=',')

#~ 		for(i in 0:99){
#~ 			cmd=sprintf('bsub -o log/%s_sclvm_%d.log -e log/%s_sclvm_%d.err -n 1 -M 20000 -R "rusage[mem=20000, tmp=3000]" python ../lib/mscLVM.py regressed/%s.csv regressed/pData_%s.csv regressed/techNoise_%s.csv regressed/%s_scLVM.csv 100 %s',norm_name,i,norm_name,i,norm_name,sample_name,sample_name,norm_name,i)
#~ 			print(cmd)
#~ 			system(cmd)
#~ 		}
}


if(opt=='fscLVM'){
	y<-read.table(sprintf("regressed/%s_fscLVM.csv",norm_name),header=T,sep=',',row.names = 1)
	saveRDS(y, file = sprintf("regressed/%s_fscLVM.rds",norm_name))
}


