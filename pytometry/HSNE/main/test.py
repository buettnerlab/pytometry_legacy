import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import time as time
import numpy as np
from scipy.sparse import csr_matrix
from pytometry.HSNE.main.hsne_prog import HSNE, tSNE

# IMPORTANT! Change this link to the location of a h5ad file
# Example:   r"/home/username/Public/myAnndataFile.h5ad"
filelocation =  r"/home/foelix/Öffentlich/VBh_converted.h5ad"


### TIMETEST ###
def timetest(subsamplepercentage=None):
    '''
    Processes the same dataset with descending subsampling
    :param subsamplepercentage --> list of subsamplepercentages
    '''
    print('TIMETEST')
    if subsamplepercentage is None:
        subsamplepercentage = [p for p in np.array(range(1,8,1))/100]
    timearr = list()
    for sp in subsamplepercentage:
        print('--- %s ---'%sp)
        adata = anndata.read_h5ad(filelocation)
        sc.pp.subsample(adata, sp)
        hsne = HSNE(adata, imp_channels=imp_channels)
        hsne.load_adata(adata, imp_channels=imp_channels)  # load adata file
        p0 = time.time()
        hsne.fit(scale=3)
        p1 = time.time()
        timearr.append(p1-p0)
    plt.plot(subsamplepercentage, timearr)
    plt.show()
#timetest()

def time_avg_test(n, subsamplepercentage):
    '''
    Processes the loaded dataset n times and shows average time
    :param n: number of iterations
    :param subsamplepercentage: percentage of subsampling
    '''
    print('TIME AVG TEST')
    timearr = list()
    for i in range(n):
        print('--- %s ---' % i)
        adata = anndata.read_h5ad(filelocation)
        sc.pp.subsample(adata, subsamplepercentage)
        hsne = HSNE(adata, imp_channels=imp_channels)
        hsne.load_adata(adata, imp_channels=imp_channels)  # load adata file
        p0 = time.time()
        hsne.fit(scale=3)
        p1 = time.time()
        timearr.append(p1 - p0)
    print("RESULT AVG TEST: \nSubsample: %s\nAvg time: %s\nEach run:%s"%(subsamplepercentage,np.mean(timearr),timearr))
#time_avg_test(3, 0.08)


### GRAPHS ###
subsampling_percentage = 0.2
adata = anndata.read_h5ad(filelocation)
sc.pp.subsample(adata, subsampling_percentage)
imp_channels = [1,3,5,7,9,13]
hsne = HSNE(adata, imp_channels=imp_channels)
hsne.load_adata(adata, imp_channels=imp_channels)  # load adata file

print('Test started...')
p0 = time.time()
scales = hsne.fit(scale=3, calc_embedding='last', verbose=True)
p1 = time.time()
print('--- Complete Test finished in %s seconds! ---'%(p1-p0))

hsne.show_selected_lm(scale_num=0)
hsne.show_selected_lm(scale_num=1)
hsne.show_selected_lm(scale_num=2)


for scale in enumerate(scales):
    hsne.plot(scale_num=scale[0])
print('Select Points')
subset_ind = hsne.lasso_subset(num=1)
print(np.shape(subset_ind))

# TODO doesnt work
t = scales[1].T[subset_ind][:, subset_ind]
t = t.multiply(csr_matrix(1.0 / np.abs(t).sum(1)))
# p = hsne.calc_T(t, scales[1].X[subset_ind])
# normalization copied from marius
p =hsne.calc_P(scales[1].T[subset_ind][:, subset_ind], scales[1].X[subset_ind])
# p = p.multiply(csr_matrix(1.0 / np.abs(p).sum(1)))

tsne = tSNE()
X_hsne = tsne.fit_transform(scales[1].X[subset_ind], p)

plt.scatter(X_hsne[:,0],X_hsne[:,1])
plt.show()

print([np.shape(s.X_hsne) for s in scales])

print('Done')
