import numpy as np
import pandas as pd
import numpy.random as npr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import sparse, stats
from scipy.special import expit
import scipy.stats
import logging

logger = logging.getLogger(__name__)

class gwas_simulated_data(object):
    # Reference:
    # https://github.com/raquelaoki/ParKCa/blob/master/src/datapreprocessing.py

    def __init__(self, n_units=10000, n_causes=100, seed=4, pca_path='data//tgp_pca2.txt', prop_tc=0.1,
                 true_causes=None, unit_test=False):
        self.n_units = n_units
        self.n_causes = n_causes
        if true_causes is None:
            self.true_causes = np.max([1, int(n_causes * prop_tc)])
        else:
            self.true_causes = true_causes
        self.confounders = self.n_causes - self.true_causes
        self.seed = seed
        self.pca_path = pca_path
        self.S = np.loadtxt(self.pca_path, delimiter=',')
        self.prop_tc = prop_tc
        self.unit_test = unit_test
        if self.unit_test:
            logging.basicConfig(level=logging.DEBUG)

        logger.debug('Dataset - GWAS initialized!')

    def generate_samples(self, prop=[0.2, 0.2, 0.05]):
        """
        Input:
        n_units, n_causes: dimentions
        snp_simulated datasets
        y: output simulated and truecases for each datset are together in a single matrix
        Note: There are options to load the data from vcf format and run the pca
        Due running time, we save the files and load from the pca.txt file
        G = X
        """
        x, y, t, tau = self.sim_dataset(self.sim_genes_TGP(D=3, prop=prop))
        y = y.reshape(self.n_units, -1)
        return x, y, t, tau

    def sim_genes_TGP(self, D, prop=[0.2, 0.2, 0.05]):
        """
        #Adapted from Deconfounder's authors
        generate the simulated data
        input:
            - Fs, ps, n_hapmapgenes: not adopted in this example
            - n_causes = integer
            - n_units = m (columns)
            - S: PCA output n x 2
        """
        np.random.seed(self.seed)
        S = expit(self.S)
        Gammamat = np.zeros((self.n_causes, 3))
        Gammamat[:, 0] = prop[0] * npr.uniform(size=self.n_causes)  # 0.2
        Gammamat[:, 1] = prop[1] * npr.uniform(size=self.n_causes)  # 0.2
        Gammamat[:, 2] = prop[2] * np.ones(self.n_causes)
        S = np.column_stack((S[npr.choice(S.shape[0], size=self.n_units, replace=True),], \
                             np.ones(self.n_units)))
        # print(S[0:5,0:5])
        F = S.dot(Gammamat.T)
        # it was 2 instead of 1: goal is make SNPs binary
        G = npr.binomial(1, F)
        # unobserved group
        lambdas = KMeans(n_clusters=3, random_state=123).fit(S).labels_
        # sG = sparse.csr_matrix(G)
        return G, lambdas

    def sim_dataset(self, G0, lambdas):
        """
        calculate the target Y based on the simulated dataset
        input:
        G0: level 0 data
        lambdas: unknown groups
        n_causes and n_units: int, dimensions of the dataset
        output:
        G: G0 in pandas format with colnames that indicate if its a cause or not
        tc: causal columns
        y01: binary target
        """
        np.random.seed(self.seed)

        tc_ = npr.normal(loc=0, scale=0.5 * 0.5, size=self.true_causes)
        tc = np.hstack((tc_, np.repeat(0.0, self.confounders)))  # True causes
        tau = stats.invgamma(3, 1).rvs(3, random_state=99)
        sigma = np.zeros(self.n_units)
        sigma = [tau[0] if lambdas[j] == 0 else sigma[j] for j in range(len(sigma))]
        sigma = [tau[1] if lambdas[j] == 1 else sigma[j] for j in range(len(sigma))]
        sigma = [tau[2] if lambdas[j] == 2 else sigma[j] for j in range(len(sigma))]
        y0 = np.array(tc).reshape(1, -1).dot(np.transpose(G0))
        l1 = lambdas.reshape(1, -1)
        y1 = (np.sqrt(np.var(y0)) / np.sqrt(0.4)) * (np.sqrt(0.4) / np.sqrt(np.var(l1))) * l1
        e = npr.normal(0, sigma, self.n_units).reshape(1, -1)
        y2 = (np.sqrt(np.var(y0)) / np.sqrt(0.4)) * (np.sqrt(0.2) / np.sqrt(np.var(e))) * e
        p = 1 / (1 + np.exp(y0 + y1 + y2))
        y01 = [npr.binomial(1, p[0][i], 1)[0] for i in range(len(p[0]))]
        y01 = np.asarray(y01)
        G, col = self.add_colnames(G0, tc)

        treatment = G.iloc[:, col[0]].values.reshape(-1)
        G.drop(col[0], axis=1, inplace=True)

        y = y0 + y1 + y2

        logger.debug('... Covariates: %i', G.shape[1] - len(col))
        logger.debug('... Target (y) : %f', np.sum(y01) / len(y01))
        logger.debug('... Sample Size: %i', G.shape[0])
        if len(col) == 1:
            T = G.iloc[:, col[0]].values
            logger.debug('... Proportion of T: %f', sum(T) / len(T))
        logger.debug('Dataset - GWAS Done!')
        return G, y, treatment, tc_

    def add_colnames(self, data, truecauses):
        """
        from matrix to pandas dataframe, adding colnames
        """
        colnames = []
        causes = 0
        noncauses = 0
        columns = []
        for i in range(len(truecauses)):
            if truecauses[i] != 0:
                colnames.append('causal_' + str(causes))
                causes += 1
                columns.append(i)
            else:
                colnames.append('noncausal_' + str(noncauses))
                noncauses += 1

        data = pd.DataFrame(data)
        data.columns = colnames
        return data, columns



class ihdp_data(object):
    # source code: https://github.com/AMLab-Amsterdam/CEVAE.git
    def __init__(self, id=1, path='/content/CEVAE/datasets/IHDP/'):
        data = pd.read_csv(path + 'ihdp_npci_' + str(id) + '.csv', sep=',', header=None)
        columns = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                   'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
                   'x23', 'x24', 'x25']
        data.columns = columns
        self.data = data
        print('IHCP initilized!')

    def generate_samples(self):
        X = self.data.drop(['y_factual', 'y_cfactual', 'mu0', 'mu1'], axis=1)
        y = self.data['y_factual'].values
        col = [0]
        tc = self.data['mu1'].mean() - self.data['mu0'].mean()
        return X, y, col, tc
