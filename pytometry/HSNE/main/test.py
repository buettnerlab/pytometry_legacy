from main.hsne_prog import HSNE, tSNE
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import time as time
import numpy as np
from scipy.sparse import csr_matrix

filelocation = r"/home/felix/Public/VBh_converted.h5ad"
adata = anndata.read_h5ad(filelocation)
sc.pp.subsample(adata, 0.2)
imp_channels = [1,3,5,7,9,13]
hsne = HSNE(adata, imp_channels=imp_channels)
hsne.load_adata(adata, imp_channels=imp_channels)  # load adata file


def timetest(subsamplepercentage=None):
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

def time_avg_test(n, subsample):
    print('TIME AVG TEST')
    timearr = list()
    for i in range(n):
        print('--- %s ---' % i)
        adata = anndata.read_h5ad(filelocation)
        sc.pp.subsample(adata, subsample)
        hsne = HSNE(adata, imp_channels=imp_channels)
        hsne.load_adata(adata, imp_channels=imp_channels)  # load adata file
        p0 = time.time()
        hsne.fit(scale=3)
        p1 = time.time()
        timearr.append(p1 - p0)
    print("RESULT AVG TEST: \nSubsample: %s\nAvg time: %s\nEach run:%s"%(subsample,np.mean(timearr),timearr))

#time_avg_test(3, 0.08)


print('Test started...')
p0 = time.time()
scales = hsne.fit(scale=4, calc_embedding='last', verbose=True)
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
X_tsne = tsne.fit_transform(scales[1].X[subset_ind], p)

plt.scatter(X_tsne[:,0],X_tsne[:,1])
plt.show()

print([np.shape(s.X_tsne) for s in scales])






'''
scale_num = 1
print('Scale number %d'%scale_num)
for channel in enumerate(hsne.adata.uns['imp_channels']):
    plt.subplot(3, 2, channel[0] + 1)
    plt.scatter(scales[scale_num].X_tsne[:, 0], scales[scale_num].X_tsne[:, 1], c=scales[scale_num].X[:, channel[0]])
    plt.title('Colored by %s' % hsne.adata.var_names[channel[1]])
    plt.colorbar()
plt.show()


hsne.T = hsne.calc_T(hsne.X, n_neighbors=20)  # calc first T
lm_layer_one = hsne.get_landmarks(hsne.X, hsne.T, beta=100, teta=50) # b 50 t 25
landmarks = adata.X[lm_layer_one]
I = hsne.calc_AoI(lm_layer_one, hsne.T)
T_s = hsne.calc_next_T(I, np.ones((np.shape(I)[0],)))
P = hsne.calc_joint_probs(T_s)#(T_s + T_s.transpose())/(2*len(landmarks))
tsne = tSNE()#_t_sne.TSNE()
X_tsne = tsne.fit_transform(landmarks, P=P)





from scramble.pyplotLasso import SelectFromCollection
fig, ax = plt.subplots()
pts = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10)
selector = SelectFromCollection(ax, pts)
def accept(event):
    if event.key == "enter":
        zoomed_lm = selector.ind
        print("Selected points indices:")
        print(selector.ind)
        selector.disconnect()
        ax.set_title("")
        fig.canvas.draw()

fig.canvas.mpl_connect("key_press_event", accept)
ax.set_title("Press enter to accept selected points.")
plt.show()


for channel in enumerate(hsne.adata.var_names[imp_channels]):
    plt.subplot(3,2,channel[0]+1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=landmarks[:, channel[0]])
    plt.title('Colored by %s' % channel[1])
    plt.colorbar()
plt.show()
'''
print('Done')