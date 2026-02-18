# Use Python3.11, e.g. module add python3.11
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text

# Plot psi and delta psi from Shea et al vs. Tanner et al.
def shea_vs_tanner(in_f, out_f, list_f):
  
    genes = [] 
    data = [] 
    for line in open(in_f):
        if line.startswith("#"):
            header = line.strip().split("\t")
        else:
            vals = line.strip().split("\t")
            gene = vals[0]
            genes.append(gene)
            data.append(list(map(float, vals[1:])))

    data = np.array(data)


    # Max delta between Tanner and emma is "t" 
    thi = .3
    tlo = .1
    mindelta = .2
    idx = np.where(\
        ((data[:, 4] - data[:, 0] <= thi) & (data[:, 0] - data[:, 4] <= tlo)) &\
        ((data[:, 6] - data[:, 2] <= thi) & (data[:, 2] - data[:, 6] <= tlo)) &\
        (abs(data[:, 0] - data[:, 2]) >= mindelta) & (abs(data[:, 4] - data[:, 6]) >= mindelta))[0]


    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.errorbar(data[:, 0], data[:, 4], xerr=data[:, 1], yerr=data[:, 5],\
        fmt='o', color='#3498db', label='WT', lw=0.5, ms=5)
    plt.errorbar(data[:, 2], data[:, 6], xerr=data[:, 3], yerr=data[:, 7],\
        fmt='o', color='#e67e22', label=r'$HSA^{\mathrm{LR}}$', lw=0.5, ms=5)

    plt.scatter(data[idx, 0], data[idx, 4], marker='o', s=25, c='#3498db', zorder=3, edgecolor='k')
    plt.scatter(data[idx, 2], data[idx, 6], marker='o', s=25, c='#e67e22', zorder=3, edgecolor='k')
 
    print(len(idx))
    genelist = open(list_f, 'w')
    texts = []
    for i in range(len(genes)):
        if i in idx:
            print(genes[i], 'YES')
            genelist.write(genes[i] + "\n")
    genelist.close()

    plt.plot([tlo, 1], [0, 1 - tlo], dashes=(2, 2), color='#CCCCCC', lw=1) 
    plt.plot([0, 1 - thi], [thi, 1], dashes=(2, 2), color='#CCCCCC', lw=1) 
    plt.fill_between([0, 1], y1=[-tlo, 1 - tlo], y2=[thi, 1 + thi], color='#DDDDDD', zorder=-1)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('$\Psi$, Tanner et al', fontsize=12) 
    plt.ylabel('$\Psi$, Shea et al', fontsize=12) 
    plt.legend(loc='lower right') 
    sns.despine() 
   
    # Plot deltas 
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0] - data[:, 2], data[:, 4] - data[:, 6], s=25, color='#AAAAAA')
    plt.scatter(data[idx, 0] - data[idx, 2], data[idx, 4] - data[idx, 6], s=25, color='#AAAAAA', edgecolor='k')
    
    plt.plot([-.2, -.2], [-1, 1], dashes=(2, 2), color='#CCCCCC', lw=1, zorder=0) 
    plt.plot([-1, 1], [-.2, -.2], dashes=(2, 2), color='#CCCCCC', lw=1, zorder=0) 
    plt.plot([.2, .2], [-1, 1], dashes=(2, 2), color='#CCCCCC', lw=1, zorder=0) 
    plt.plot([-1, 1], [.2, .2], dashes=(2, 2), color='#CCCCCC', lw=1, zorder=0) 
    plt.fill_between([-1, -mindelta], y1=[-1, -1], y2=[-mindelta, -mindelta], color='#DDDDDD', zorder=-1)
    plt.fill_between([mindelta, 1], y1=[mindelta, mindelta], y2=[1, 1], color='#DDDDDD', zorder=-1)
    
    plt.xlabel('$\Delta\Psi$, WT - $HSA^{\mathrm{LR}}$ (Tanner)', fontsize=12) 
    plt.ylabel('$\Delta\Psi$, WT - $HSA^{\mathrm{LR}}$ (Shea)', fontsize=12) 
    plt.xlim(-0.75, 0.75)
    plt.ylim(-0.75, 0.75)
    sns.despine() 
   
    texts = [] 
    for i in range(len(genes)):
        texts.append(plt.text(data[i, 0] - data[i, 2], data[i, 4] - data[i, 6], genes[i], size=8))
    
    adjust_text(texts, expand=(2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

    plt.savefig(out_f) 


# Filter by mean squared error from the time course.
def filter_by_mse(in_f, out_f, plot_f):

    genes = []
    psi = []
    for line in open(in_f):
        if line.startswith("#geneSymbol"):
            header1 = line.strip().split("\t")
        elif line.startswith("#Group"):
            header2 = line.strip().split("\t")
        else:
            vals = line.strip().split("\t")
            gene = vals[0]
            genes.append(gene)
            tmp = vals[5:29]
            for i in range(len(tmp)):
                try:
                    tmp[i] = float(tmp[i])
                except:
                    pass
            psi.append(tmp)
                
    t = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 7, 7, 7, 7])
    
    import numpy.ma as ma
    psi = np.array(psi, dtype=object)
    psi = ma.masked_values(psi, 'NA')
    psi = psi.filled(fill_value=np.nan)
    psi = psi.T
    print(psi.shape)

    import operator
    out = open(out_f, 'w')
    errors = []
    for i in range(len(genes)):
        allerrs = []
        for time in [-1, 0, 1, 2, 4, 7]:
            idx = np.where(t == time)[0]
            delta = (psi[idx, i] - psi[idx, i].mean())
            errs = [x**2 for x in delta]
            allerrs.extend(errs)
        mse = np.nanmean(allerrs)
        errors.append([genes[i], mse])
        print(genes[i], mse)
        if mse < 0.006:
            out.write(genes[i] + "\n")
    out.close()

    errors.sort(key=operator.itemgetter(1))
    plt.figure(figsize=(6, 1.5))
    plt.bar([x[0] for x in errors], [x[1] for x in errors])
    plt.subplots_adjust(left=.1, bottom=.3)
    plt.ylabel('Mean squared error', fontsize=6)
    plt.xticks(fontsize=6, rotation=45, ha='right')
    plt.yticks(np.linspace(0, .025, 6), fontsize=6)
    plt.axhline(y=0.006, dashes=(8, 8), color='k', lw=.5)
    sns.despine()
    plt.savefig(plot_f)
     
def overlap_lists(list1_f, list2_f, out_f):
   
    events1 = []
    events2 = [] 
    for line in open(list1_f):
        events1.append(line.strip())
    for line in open(list2_f):
        events2.append(line.strip())

    out = open(out_f, 'w')
    out.write("\n".join([x for x in events1 if x in events2]))
    out.close()
