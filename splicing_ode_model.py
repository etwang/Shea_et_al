# Use Python3.11, e.g. module add python3.11
import os, sys
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as ss
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from pytensor.compile.ops import as_op
from pymc import Slice, HalfCauchy, Model, Normal, Uniform, HalfNormal,Truncated, TruncatedNormal, Deterministic, DiscreteUniform, sample, math
from adjustText import adjust_text


# Define the system of ODEs
# MBNL changes as a function of time
# Nascent psi changes with MBNL
# Observed psi changes with nascent psi over time and depends on rate of replacement
def fullmodel(y, t, a, b, k, psimin, psimax, ec50, slope):

    nevents = len(psimin)
    MBNL, Npsi, Opsi = y.reshape(-1, nevents) # reshape the single vector into 2x2 array

    # Differential equations
    dMBNL_dt = a - b * MBNL
    dNpsi_dMBNL = (psimax - psimin) * slope * np.log(10) * 10**(slope * (ec50 - MBNL)) /\
        (1 + 10**(slope * (ec50 - MBNL)))**2
    dNpsi_dt = dNpsi_dMBNL * dMBNL_dt
    dOpsi_dt = k * Npsi - k * Opsi
    sol = np.stack([dMBNL_dt, dNpsi_dt, dOpsi_dt]).flatten() # pack and flatten the data
    return sol

# Model for how Psi changes with MBNL
def model2(psi, mbnl, psimin, psimax, ec50, slope):

    # Differential equations
    dpsi_dMBNL = (psimax - psimin) * slope * np.log(10) * 10**(slope * (ec50 - mbnl)) /\
        (1 + 10**(slope * (ec50 - mbnl)))**2
    return dpsi_dMBNL

    
# Helper function to read psi values and selected events
def readData(in_f, selected_f):

    includelist = []
    for line in open(selected_f):
        includelist.append(line.strip())

    genes = []
    psimin = []
    psimax = []
    ec50 = []
    slope = []
    obs_psi = []
    for line in open(in_f):
        if line.startswith("#geneSymbol"):
            header1 = line.strip().split("\t")
        elif line.startswith("#Group"):
            header2 = line.strip().split("\t")
        else:
            vals = line.strip().split("\t")
            gene = vals[0]
            params = list(map(float, vals[1:5]))
            if gene in includelist:
                genes.append(gene)
                psimin.append(params[0])
                psimax.append(params[1])
                ec50.append(params[2])
                slope.append(params[3])
                psivals = list(map(float, vals[5:]))
                obs_psi.append(psivals) 

    psimin = np.array(psimin)
    psimax = np.array(psimax)
    ec50 = np.array(ec50)
    slope = np.array(slope)
    obs_psi = np.array(obs_psi).T
    obs_psi[np.where(obs_psi == -1)] = np.nan
    return genes, obs_psi, psimin, psimax, ec50, slope

# Main script to infer a, b, and k
def infer(in_f, selected_f, out_f, niter):

    niter = int(niter)
 
    genes, Opsi_, psimin, psimax, ec50, slope = readData(in_f, selected_f)
    for i in range(len(genes)):
        print(i, genes[i])

    t_ = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 7, 7, 7, 7])
    Opsi_ = Opsi_[4:24, :]


    MBNL0 = .45 # estimated using all splicing events 

    # Set up bayesian inference model
    nevents = Opsi_.shape[1]

    with Model() as model:

        a_var = Uniform('a', 0, 1)
        b_var = Uniform('b', a_var, 1)
        k_var = TruncatedNormal('k', mu=1, sigma=10, lower=0, shape=nevents)
        sigma = Uniform('sigma', 0, 1)
        
        # Solve ODE
        @as_op(itypes=[pt.dscalar, pt.dscalar, pt.dvector], otypes=[pt.dmatrix, pt.dmatrix, pt.dmatrix])
        def ode_solver(a, b, k):
            MBNL_init = np.asarray([MBNL0 for i in range(nevents)])
            Npsi_init = psimin + (psimax - psimin) / (1 + 10**(slope * (ec50 - MBNL0)))
            Opsi_init = Npsi_init
            y0 = np.stack([MBNL_init, Npsi_init, Opsi_init])
            y0 = y0.flatten()
            solution = odeint(fullmodel, y0, t_, args=(a, b, k, psimin, psimax, ec50, slope))
            MBNL, Npsi, Opsi = np.split(solution, 3, axis=1)
            return [MBNL, Npsi, Opsi]

        MBNL, Npsi, Opsi = ode_solver(a_var, b_var, k_var)
        print(MBNL.eval().shape)
        likelihood = Normal("y", mu=Opsi, sigma=sigma, observed=Opsi_) 
    
    print("Set up model.")

    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:-1]

    # Specify the sampler
    sampler = "Slice Sampler"
    tune = draws = int(niter) 

    # Inference
    with model:
        trace = pm.sample(step=[pm.Slice(vars_list)], tune=tune, draws=draws)
    trace.to_netcdf(out_f)

# Compare K values to nascent/total values
def k_vs_nascent(nascent_f, psi_f, selected_f, db_f, out_f):
    
    genes, psi, psimin, psimax, ec50, slope = readData(psi_f, selected_f)
    
    t_ = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 7, 7, 7, 7])
    psi = psi[4:24, :]
    
    trace = az.from_netcdf(db_f)

    a = getMax(trace.posterior["a"].values.flatten(), 0, 1, 100)
    b = getMax(trace.posterior["b"].values.flatten(), 0, 1, 100)
   
    k = [] 
    for i in range(len(genes)):
        k.append(getMax(trace.posterior["k"][:, :, i].values.flatten(), 0, 40, 4000))

    # Obtain curves for MBNL, Npsi, Opsi
    MBNL0 = [0.44 for i in range(len(psimin))]
    Npsi0 = psimin + (psimax - psimin) / (1 + 10**(slope * (ec50 - MBNL0)))
    Opsi0 = Npsi0
    
    # Solve ODE
    y0 = np.stack([MBNL0, Npsi0, Opsi0])
    y0 = y0.flatten()
    solution = odeint(fullmodel, y0, t_, args=(a, b, k, psimin, psimax, ec50, slope))

    # Extract solutions
    # Dimensions of Npsi are time x gene 
    MBNL, Npsi, Opsi = np.split(solution, 3, axis=1)
    
    geneToTpm = {} 
    for line in open(nascent_f):
        if line.startswith("#"):
            header = line.strip().split("\t")[1:]
            headerToIdx = {}
            for i in range(len(header)):
                headerToIdx[header[i]] = i
        else:
            vals = line.strip().split("\t")
            gene = vals[0]
            if gene in genes:
                tpm = list(map(float, vals[1:]))
                # Indices are total1: 0:3, total2: 3:6, total0: 6:10
                # nascent1: 10:13, nascent2: 13:16, nascent0: 16:20
                geneToTpm[gene] = [np.mean(tpm[6:10]), np.mean(tpm[0:3]), np.mean(tpm[3:6]),\
                    np.mean(tpm[16:20]), np.mean(tpm[10:13]), np.mean(tpm[13:16])]
    print(geneToTpm.keys())

    geneToK = outputK(psi_f, selected_f, db_f)
  
    data = [] 
    labels = []
    allerrors = []
    for i in range(len(genes)):
        gene = genes[i]
        k = geneToK[gene]
        tpm = geneToTpm[gene]
 
        # Compute MSE of the fit to the data
        err = []
        for j in range(len(t_)):
            err.append((Opsi[j, i] - psi[j, i])**2)
        err = np.mean(err)
        allerrors.append(err)
 
        print(gene, k[0], err) 
        labels.append(gene)
        nascent = tpm[5]
        total = tpm[2]
        data.append([k[0], nascent / total, err])

    allerrors = np.array(allerrors)
    data = np.array(data)

    idx = np.where(data[:, 2] < 0.01)[0]
    print(np.corrcoef(np.log(data[:, 0]), np.log(data[:, 1]))[0][1])
    print(np.corrcoef(np.log(data[idx, 0]), np.log(data[idx, 1]))[0][1])

    from scipy.stats import pearsonr, spearmanr
    print(pearsonr(np.log(data[:, 0]), np.log(data[:, 1])))
    print(spearmanr(np.log(data[:, 0]), np.log(data[:, 1])))

    plt.figure(figsize=(6, 4)) 
    plt.scatter(np.log(data[:, 0])/np.log(10), np.log(data[:, 1])/np.log(10), s=.3/allerrors)
    texts = []
    for i in range(len(labels)):
        texts.append(plt.text(np.log(data[i, 0])/np.log(10), np.log(data[i, 1])/np.log(10),\
            labels[i], size=8, ha='center', va='center'))
    adjust_text(texts, expand=(2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

    plt.xlabel("$log_{10}(k)$")
    plt.ylabel("$log_{10}(Nascent/Total)$")
    plt.subplots_adjust(left=.2, bottom=.2)
    sns.despine()
    plt.savefig(out_f) 
  

# Output inferred k values
def outputK(psi_f, selected_f, db_f):

    genes, psi, psimin, psimax, ec50, slope = readData(psi_f, selected_f)
    trace = az.from_netcdf(db_f)
  
    geneToK = {} 
    for i in range(len(genes)):
        maxval = getMax(trace.posterior["k"][:, :, i].values.flatten(), 0, 40, 4000)
        low = np.round(np.percentile(trace.posterior["k"][:, :, i].values.flatten(), 5), 2)
        hi = np.round(np.percentile(trace.posterior["k"][:, :, i].values.flatten(), 95), 2)
        geneToK[genes[i]] = [maxval, low, hi]
    return geneToK


# Output inferred k values
def outputK_Tanner(psi_f, db_f):

    genes, psi, psimin, psimax, ec50, slope = readData_Tanner(psi_f)
    trace = az.from_netcdf(db_f)
  
    geneToK = {} 
    for i in range(len(genes)):
        maxval = getMax(trace.posterior["k"][:, :, i].values.flatten(), 0, 40, 4000)
        low = np.round(np.percentile(trace.posterior["k"][:, :, i].values.flatten(), 5), 2)
        hi = np.round(np.percentile(trace.posterior["k"][:, :, i].values.flatten(), 95), 2)
        geneToK[genes[i]] = [maxval, low, hi]
    return geneToK


# Read Table S5 from Tanner. Does not include Brd2.
def readData_Tanner(in_f):
   
    genes = []
    psimin = []
    psimax = []
    ec50 = []
    slope = []
    obs_psi = []
    for line in open(in_f):
        print(line)
        if line.startswith("#"):
            header = line.strip().split("\t")
        else:
            vals = line.strip().split("\t")
            gene = vals[0]
            params = list(map(float, vals[1:5]))
            genes.append(gene)
            psimin.append(params[0])
            psimax.append(params[1])
            ec50.append(params[2])
            slope.append(params[3])
            psivals = list(map(float, vals[5:]))
            obs_psi.append(psivals) 

    psimin = np.array(psimin)
    psimax = np.array(psimax)
    ec50 = np.array(ec50)
    slope = np.array(slope)
    obs_psi = np.array(obs_psi).T
    return genes, obs_psi, psimin, psimax, ec50, slope


def infer_Tanner(in_f, out_f, niter):

    niter = int(niter)
 
    genes, Opsi_, psimin, psimax, ec50, slope = readData_Tanner(in_f)
    for i in range(len(genes)):
        print(i, genes[i])

    t_ = np.array([0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 7, 7, 7, 10, 10, 10,\
        14, 14, 14, 17, 17, 17, 21, 21, 21, 28, 28, 28])
    Opsi_ = Opsi_

    MBNL0 = .404 # estimated using all ctl_ASO samples
 
    # Set up bayesian inference model
    nevents = Opsi_.shape[1]

    with Model() as model:

        a_var = Uniform('a', 0, 1)
        b_var = Uniform('b', a_var, 1)
        k_var = TruncatedNormal('k', mu=1, sigma=10, lower=0, shape=nevents)
        sigma = Uniform('sigma', 0, 1)
        
        # Solve ODE
        @as_op(itypes=[pt.dscalar, pt.dscalar, pt.dvector], otypes=[pt.dmatrix, pt.dmatrix, pt.dmatrix])
        def ode_solver(a, b, k):
            MBNL_init = np.asarray([MBNL0 for i in range(nevents)])
            Npsi_init = psimin + (psimax - psimin) / (1 + 10**(slope * (ec50 - MBNL0)))
            Opsi_init = Npsi_init
            y0 = np.stack([MBNL_init, Npsi_init, Opsi_init])
            y0 = y0.flatten()
            solution = odeint(fullmodel, y0, t_, args=(a, b, k, psimin, psimax, ec50, slope))
            MBNL, Npsi, Opsi = np.split(solution, 3, axis=1)
            return [MBNL, Npsi, Opsi]

        MBNL, Npsi, Opsi = ode_solver(a_var, b_var, k_var)
        print(MBNL.eval().shape)
        likelihood = Normal("y", mu=Opsi, sigma=sigma, observed=Opsi_) 
    
    print("Set up model.")

    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:-1]

    # Specify the sampler
    sampler = "Slice Sampler"
    tune = draws = int(niter) 

    # Inference
    with model:
        trace = pm.sample(step=[pm.Slice(vars_list)], tune=tune, draws=draws)
    trace.to_netcdf(out_f)


# This has been the most common use case:
def plot_results_with_psi(psi_f, selected_f, in_f, out_f, params_f):
  
    params = open(params_f, 'w') 
    genes, psi, psimin, psimax, ec50, slope = readData(psi_f, selected_f)
    t = np.linspace(0, 10, 100)
    
    t_ = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 7, 7, 7, 7])
    psi = psi[4:24, :]
    
    trace = az.from_netcdf(in_f)

    a = getMax(trace.posterior["a"].values.flatten(), 0, 1, 100)
    b = getMax(trace.posterior["b"].values.flatten(), 0, 1, 100)
    
    params.write("a\t" + str(a) + "\n")   
    params.write("b\t" + str(b) + "\n")   
   
    k = [] 
    for i in range(len(genes)):
        thisk = getMax(trace.posterior["k"][:, :, i].values.flatten(), 0, 40, 1000)
        k.append(thisk)
        params.write(genes[i] + "\t" + str(thisk) + "\n")   

    params.write(az.summary(trace).to_string())
    params.close()

    print(az.summary(trace))

    # Obtain curves for MBNL, Npsi, Opsi
    MBNL0 = [0.45 for i in range(len(psimin))]
    Npsi0 = psimin + (psimax - psimin) / (1 + 10**(slope * (ec50 - MBNL0)))
    Opsi0 = Npsi0
    
    # Time points where solution is computed
    t = np.linspace(0, 10, 100)  # from time 0 to 10
    
    # Solve ODE
    y0 = np.stack([MBNL0, Npsi0, Opsi0])
    y0 = y0.flatten()
    solution = odeint(fullmodel, y0, t, args=(a, b, k, psimin, psimax, ec50, slope))

    # Extract solutions
    MBNL, Npsi, Opsi = np.split(solution, 3, axis=1)

    # Plot distributions of each parameter.
    # For MBNL, plot the estimated MBNL as a function of time, with shaded
    # confidence intervals.
    # For splicing events, Also plot estimated nascent and observed
    # curves for each event across time.
    plt.figure(figsize=(12, len(genes) * 2))

    plt.subplot2grid((len(genes) + 1, 3), (0, 0))
    sns.kdeplot(trace.posterior["a"].values.flatten())
    plt.xlim(0, 1)
    plt.title("a")
    
    plt.subplot2grid((len(genes) + 1, 3), (0, 1))
    sns.kdeplot(trace.posterior["b"].values.flatten())
    plt.xlim(0, 1)
    plt.title("b")
    
    plt.subplot2grid((len(genes) + 1, 3), (0, 2))
    plt.plot(t, MBNL)     

    for i in range(len(t)):
        print(t[i], MBNL[i])

    plt.xlabel('Time')
    plt.ylabel('[MBNL]')
    plt.ylim(0, 1)


    # Model for Psi as a function of MBNL
    yinit = psimin + (psimax - psimin) / (1 + 10**(slope * ec50))
    psi_dmbnl = odeint(model2, y0=yinit,\
        t=np.linspace(0, 1, 100), args=(psimin, psimax, ec50, slope))

    for i in range(len(genes)):
        plt.subplot2grid((len(genes) + 1, 3), (i + 1, 0))
        sns.kdeplot(trace.posterior["k"][:, :, i].values.flatten())
        plt.axvline(trace.posterior["k"][:, :, i].values.flatten().mean(),\
            color='r', dashes=(2, 2))
        plt.axvline(np.percentile(trace.posterior["k"][:, :, i].values.flatten(), 50),\
            color='k', dashes=(2, 2))
        plt.axvline(getMax(trace.posterior["k"][:, :, i].values.flatten(), 0, 40, 100),\
            color='g', dashes=(2, 2))
        plt.xlim(0, 30)
        plt.xlabel('k')
        plt.title(genes[i])
    
        plt.subplot2grid((len(genes) + 1, 3), (i + 1, 1))
        plt.plot(t, Npsi[:, i], color='#CCCCCC')
        plt.plot(t, Opsi[:, i])
        plt.scatter(t_, psi[:, i], s=8, color='r', zorder=10)
        plt.xlabel('Time')
        plt.ylabel('Psi')
        plt.xlim(0, 10)
        plt.ylim(0, 1)
        plt.title(genes[i])

        plt.subplot2grid((len(genes) + 1, 3), (i + 1, 2))
        plt.plot(np.linspace(0, 1, 100), psi_dmbnl[:, i], label=\
             'min:%s, max:%s, ec50:%.2f, slope:%.2f'%\
            (psimin[i], psimax[i], ec50[i], slope[i]))
        plt.title(genes[i])
        plt.xlabel('[MBNL]')
        plt.ylabel('Psi')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    sns.despine()
    #az.rcParams["plot.max_subplots"] = 100
    #az.plot_trace(trace, compact= False, legend = True)
    plt.subplots_adjust(hspace=1, wspace=.5)
    #az.summary(trace, round_to=2).to_csv('timecourse summary', sep = '\t')
    plt.savefig(out_f)


# Plot K values from Shea et al vs. Tanner et al
def shea_vs_tanner(sheapsi_f, selected_f, sheadb_f, tannerpsi_f, tannerdb_f, out_f):
    
    sheaToK = outputK(sheapsi_f, selected_f, sheadb_f)
    tannerToK = outputK_Tanner(tannerpsi_f, tannerdb_f)

    data = []
    genes = []
    for gene in sheaToK:
        genes.append(gene)
        data.append([sheaToK[gene][0], tannerToK[gene][0]])
        print(gene, sheaToK[gene], tannerToK[gene])

    data = np.array(data)
   
    plt.figure(figsize=(6, 4)) 
    plt.scatter(np.log(data[:, 0]) / np.log(10), np.log(data[:, 1]) / np.log(10))

    print(np.corrcoef(np.log(data[:, 0]), np.log(data[:, 1]))[0][1])
    texts = []
    for i in range(len(genes)):
        texts.append(plt.text(np.log(data[i, 0])/np.log(10), np.log(data[i, 1])/np.log(10),\
            genes[i], size=8, ha='center', va='center'))
    adjust_text(texts, expand=(2, 2), arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    plt.xlabel("$log_{10}(k)$, Shea et al")
    plt.ylabel("$log_{10}(k)$, Tanner et al")
    sns.despine()
    plt.savefig(out_f)

# Get value at max of distribution
def getMax(vec, low, high, n):
    kde = gaussian_kde(vec)
    xvals = np.linspace(low, high, n)
    kde_vals = kde(xvals)
    return xvals[np.argmax(kde_vals)]

# Output percentile values for each inferred parameter
def output_percentiles(psi_f, selected_f, db_f, out_f):
   
    genes, Opsi_, psimin, psimax, ec50, slope = readData(psi_f, selected_f)

    params = open(out_f, 'w')
 
    trace = az.from_netcdf(db_f)

    params.write("#Parameter\tValAtMax")
    perc = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    for i in perc:
        params.write("\t" + str(i))
    params.write("\n")
    
    avals = [getMax(trace.posterior["a"].values.flatten(), 0, 1, 100)]
    for i in perc:
        avals.append(np.percentile(trace.posterior["a"].values.flatten(), i))
    params.write("a\t" + "\t".join(list(map(str, avals))) + "\n")

    bvals = [getMax(trace.posterior["b"].values.flatten(), 0, 1, 100)]
    for i in perc:
        bvals.append(np.percentile(trace.posterior["b"].values.flatten(), i))
    params.write("b\t" + "\t".join(list(map(str, bvals))) + "\n")
    
    for i in range(len(genes)):
        kvals = [getMax(trace.posterior["k"][:, :, i].values.flatten(), 0, 40, 1000)]
        for j in perc:
            kvals.append(np.percentile(trace.posterior["k"][:, :, i].values.flatten(), j))
        params.write(genes[i] + "\t" + "\t".join(list(map(str, kvals))) + "\n")

    params.close()

# Output percentile values for each inferred parameter
def output_percentiles_Tanner(psi_f, db_f, out_f):
   
    genes, psi, psimin, psimax, ec50, slope = readData_Tanner(psi_f)

    params = open(out_f, 'w')
 
    trace = az.from_netcdf(db_f)

    params.write("#Parameter\tValAtMax")
    perc = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    for i in perc:
        params.write("\t" + str(i))
    params.write("\n")
    
    avals = [getMax(trace.posterior["a"].values.flatten(), 0, 1, 100)]
    for i in perc:
        avals.append(np.percentile(trace.posterior["a"].values.flatten(), i))
    params.write("a\t" + "\t".join(list(map(str, avals))) + "\n")

    bvals = [getMax(trace.posterior["b"].values.flatten(), 0, 1, 100)]
    for i in perc:
        bvals.append(np.percentile(trace.posterior["b"].values.flatten(), i))
    params.write("b\t" + "\t".join(list(map(str, bvals))) + "\n")
    
    for i in range(len(genes)):
        kvals = [getMax(trace.posterior["k"][:, :, i].values.flatten(), 0, 40, 1000)]
        for j in perc:
            kvals.append(np.percentile(trace.posterior["k"][:, :, i].values.flatten(), j))
        params.write(genes[i] + "\t" + "\t".join(list(map(str, kvals))) + "\n")

    params.close()

