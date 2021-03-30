"""
Data Pre-processing from GoPDX project
Note: unfortunately, this data is not publicly available.

Description:
I used the pandas-plink library to load the data out of bed, bim, and fam files. I needed all the data to be
in a tabular format, but this pandas-plink does not provide an easy mechanism to get that. So, I had to go over
the each sample (row) and retrieve the genetic' (columns) information iteratively. Because the retrieval for
each row was very time consuming, reading the whole data this way could take hours (or days). So, to make it
faster to load the data for future use, once I read the data for a row (as a numpy array), I saved the numpy
array as a pickle file on the disk. So, for future use, all I need was to read those pickle file and construct
my final numpy array. Reading the pickle file is very fast, and the whole data can be loaded in few minutes

Example:
    goPDX()
"""

from pandas_plink import read_plink1_bin
import os
import numpy as np
import pandas as pd
import time
import allel
from scipy.spatial.distance import squareform
import operator
import functools
from scipy.sparse import coo_matrix, vstack
from scipy import sparse

# TODO: Remove these paths
#path_input = path_raw = "Cisplatin-induced_ototoxicity"
#path_output = ""
#path_output_simulated = "CompBioAndSimulated_Datasets\\data\\"
#known_snps_path = 'CompBioAndSimulated_Datasets\\data\\known_snps.txt'
#ks = pd.read_csv(known_snps_path, header=None)[0].values


class goPDX:
    """
    NOTES:
    Working only with dominant encoding
    read files from npz
    """

    def __init__(self, path="/content/", clinicalName='clinical_01Mar2021.xlsx',
                 tag='snpsback_', save_files=False, read_raw = False, path_raw = ''):
        super(goPDX, self).__init__()
        self.read_raw = read_raw
        self.path_raw = path_raw
        if self.read_raw:
            load_raw_bed()



        self.path = path
        self.clinicalName = clinicalName
        self.tag = tag
        clinical, self.data, self.variants_names, self.cadd_score, self.samples_names, self.known_snp = \
            self.loading_preprocessed(save_files)
        self.y, self.clinical, self.clinical_names = self.loading_clinical(clinical)
        saving_preprocessed(save_files)


    def loading_preprocessed(self, save_files=False):

        # Loading files
        try:
            clinical = pd.read_excel(self.clinicalName)  # new data set
        except FileNotFoundError:
            print('ERROR: ', self.clinicalName, ' missing from Colab')

        try:
            data = sparse.load_npz(self.path + self.tag + 'gt_dominant.npz')  # gt following snps and variants order
            variants_names = np.load(self.path + self.tag + 'variants.npy')  # variants names
            cadd_score = np.load(self.path + self.tag + 'cadd.npy')  # cadd score following the variants order
            samples_names = np.load(self.path + 'samples.npy', allow_pickle=True)  # samples order
            known_snps = pd.read_csv(self.path + 'known_snps_fullname.txt', header=None)[0].values  # known snps
        except FileNotFoundError:
            print('ERRROR: gt_dominant,samples, cadd, variants or known_snps_fullname missing from Colab')

        clinical = self.clinical_filtering(clinical)
        data, samples_names = self.update_data_after_clinical_filtering(data, samples_names, clinical_)
        ks_bool_ = self.ks_bool(known_snps, variants_names)
        data, variants_names, cadd_score, ks_bool_ = eliminate_low_incidence(data, variants_names, cadd_score, ks_bool_)
        data, variants_names, cadd_score, ks_bool_ = eliminate_low_cadd(data, variants_names, cadd_score, ks_bool_)

        return clinical, data, variants_names, cadd_score, samples_names, known_snps

    def loading_clinical(self, clinical, samples_names):
        clinical['ID'] = pd.Categorical(clinical['ID'], categories=self.samples_names, ordered=True)
        clinical = clinical.sort_values(by=['ID'])
        clinical.reset_index(inplace=True, drop=True)
        y = clinical['y']
        clinical = clinical.drop(["CIO_Grade", 'y', 'ID'], axis=1)
        clinical_names = clinical.columns
        return y, clinical, clinical_names

    def saving_preprocessed(self):
        # From Data pre-processing functions:
        if save_files:
            print('Saving files....')
            np.save('y', self.y)
            np.save('x_clinical', self.clinical.values)
            np.save('x_snps', self.data)
            np.save('x_colnames', self.variants_names)
            np.save('x_clinical_names', self.clinical_names)

    def ks_bool(self, known_snps, variants_name):
        """
        Create an bool array with Known SNPS (ks) positions
        :param ks: known snps (67)
        :param variants_name: variants names
        :return: bool
        """
        ks_f = [1 if v in known_snps else 0 for v in variants_name]
        ks_f = list(map(bool, ks_f))
        ks_f = np.array(ks_f)
        return ks_f

    def clinical_filtering(self, clinical):
        """
        Filter Clinical Variables (New)
        :param clinical: pd.Dataframe with clinical variables
        :return: pd.Dataframe Clinical Varibles in clean format
        """
        # Filter 1: Remove CIO_Grade Exclude and 1
        print('Original shape: ', clinical.shape)
        clinical_ = clinical[clinical.CIO_Grade != 'Exclude']
        clinical_ = clinical_[clinical_.CIO_Grade != 1]
        print('# Filter 1: new shape: ', clinical_.shape)

        # Create Y: CIO grade: 0 -> 0, 2-4 -> 1 (hearing loss happened)
        # clinical_ = clinical_.astype({'CIO_Grade': 'int32'})
        clinical_['y'] = [0 if item == 0 else 1 for item in clinical_.CIO_Grade]
        # fillna columns with NAN values (from 0 days or 0 doses)
        clinical_.fillna(0, inplace=True)

        # x days -> x
        columns_mix_numbers_string = ['VancomycinConcomitantDuration (days)', 'TobramycinConcomitantDuration (days)',
                                      'GentamicinConcomitantDuration (days)', 'AmikacinConcomitantDuration (days)',
                                      'FurosemideConcomitantDuration (days)', 'CarboplatinExactDuration (days)',
                                      'CisplatinExactDuration (days)', 'CarboplatinDose_cumulative(mg/m2)']
        for column in columns_mix_numbers_string:
            clinical_[column] = [str(item) + ' days' if str(item).isdigit() else item for item in clinical_[column]]
            clinical_[column] = clinical_[column].str.extract('(\d+)')

        # categorical variables to binary
        clinical_['Otoprotectant_Given'] = [0 if item == 'Not given' else 1 for item in
                                            clinical_['Otoprotectant_Given']]
        clinical_['Array'] = [0 if item == 'Omni' else 1 for item in clinical_['Array']]
        clinical_ = clinical_.dropna()
        print('# Filter 2: final shape', clinical_.shape)
        return clinical_

    def update_data_after_clinical_filtering(self, data, samples_names, clinical):
        remove = []
        for i in range(len(samples_names)):
            if samples_names[i] not in clinical.ID.values:
                remove.append(i)

        data_ = data.todense().transpose()
        data_ = np.delete(data_, remove, axis=0)
        samples_names = np.delete(samples_names, remove, axis=0)
        return data_, samples_names

    def raw_cadd(self):
        print('LOADING CADD')
        # os.chdir(path_input)
        cadd = pd.read_csv(self.path_raw+'Peds_CIO_merged_qc_CADD.txt', sep=" ")
        cadd['variants'] = cadd['#CHROM'].astype(str) + "_" + cadd['ID']
        return cadd.iloc[:, [6, 5]]

    def load_raw_bed(self):
        print('WARNING: THESE FUNCTIONS ARE VERY TIME CONSUMING. IT MIGHT TAKE UP TO 48 h')
        # read parts of bed values https://github.com/limix/pandas-plink/blob/master/doc/usage.rst
        # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        """Loading Files"""
        # os.chdir(path_input)
        G = read_plink1_bin(self.path_raw+"Peds_CIO_merged_qc_data.bed", bim=None, fam=None, verbose=False)
        samples = G.sample.values  # samples
        variants = G.variant.values
        s, v = len(samples), len(variants)
        print('Original shape: ', s, v)  # Shape:  454 6726287
        cadd = self.raw_cadd()

        '''Saving samples output'''
        np.save(self.path + 'samples', samples)

        '''Making sure the Cadd and variants are in the same order (very important)'''
        cadd['variants_cat'] = pd.Categorical(cadd['variants'], categories=variants, ordered=True)
        cadd_sort = cadd.sort_values(by=['variants_cat'])
        cadd_sort.reset_index(inplace=True)
        if np.equal(cadd_sort.variants, variants).sum() == len(variants):
            print('CADD and variants are in the same order')
            del cadd
        else:
            print('ERROR: CADD and variantres are in DIFFERENT order')

        cadd_sort.fillna(value={'CADD_PHRED': 0}, inplace=True)

        """First PRUNE: IF 0 IN ONE AND 1 ON ANOTHER: 2, IF 1 AND 0: 0, IF 0 AND 0: 1"""
        # Takes 48 hours to finish
        data_d, data_s, variants_, cadd_ = self.filteringSNPS(variants, cadd_sort.CADD_PHRED.values, samples, G, 'f1')
        """FINAL PRUNE: IF 0 IN ONE AND 1 ON ANOTHER: 2, IF 1 AND 0: 0, IF 0 AND 0: 1"""
        data_d, data_s, variants_, cadd_ = self.filteringSNPS(np.array(variants_), np.array(cadd_), samples, G, '', 0.2)
        fixing_erros()
        adding_known_snps_back(G, samples, variants_, cadd_, cadd_sort, data_d, data_s)


    def filteringSNPS(self, variants, cadd_sorted, samples, G, name, thold=0.05, interval=10000):
        """preparing variables"""
        variants_done = []
        cadd_done = []
        start0 = start = time.time()
        ind = 0
        build = True

        """Filtering full dataset"""
        print(range(len(variants) // interval))
        for var in range(len(variants) // interval):
            # use sparse format to reduce size
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html
            row = G.sel(sample=samples, variant=variants[ind:ind + interval]).values.transpose()

            # Counting Frequencies
            row = pd.DataFrame(row)
            row.fillna(-1, inplace=True)
            row_, variants_, cadd_ = ld_prune(row.values, variants[ind:ind + interval], cadd_sorted[ind:ind + interval],
                                              thold)  # 600 , 200, 0.05, 3
            del row
            variants_done.append(variants_)
            cadd_done.append(cadd_)
            ind = ind + interval
            row_d = np.where(row_, 0, 1)  # Dominant: 0 are 1; 1 and 2 are 0
            row_s = np.where(row_, 2, 1)  # Recessive: 2 are 1, 1 and 0 are 0 #WRONG
            # Correct filter for Resessive:np.where(test > 1, 1, 0)

            if build:
                # dominant coding:
                data_d = coo_matrix(row_d)
                # recessive coding:
                data_s = coo_matrix(row_s)
                build = False
                del row_d, row_s, row_
            else:
                data_d = vstack([data_d, coo_matrix(row_d)])
                data_s = vstack([data_s, coo_matrix(row_s)])

            if var % 10 == 0:
                print('Progress: ', round(var * 100 / (len(variants) // interval), 4), '% ---- Time Dif (s): ',
                      round(time.time() - start, 2))
                sparse.save_npz(self.path + name + 'gt_dominant.npz', data_d)
                sparse.save_npz(self.path + name + 'gt_recessive.npz', data_s)
                np.save(self.path + name + 'variants', variants_done)
                np.save(self.path + name + 'cadd', cadd_done)
                start = time.time()

        row = G.sel(sample=samples, variant=variants[ind:len(variants)]).values.transpose()
        row = pd.DataFrame(row)
        row.fillna(-1, inplace=True)

        row_, variants_, cadd_ = ld_prune(row.values, variants[ind:ind + interval], cadd_sorted[ind:ind + interval],
                                          thold)  # 600 , 200, 0.05, 3
        variants_done.append(variants_)
        cadd_done.append(cadd_)
        row_d = np.where(row_, 0, 1)
        row_s = np.where(row_, 2, 1)
        del row_

        data_d = vstack([data_d, coo_matrix(row_d)])
        data_s = vstack([data_s, coo_matrix(row_s)])

        variants_done = [item for sublist in variants_done for item in sublist]
        cadd_done = [item for sublist in cadd_done for item in sublist]

        sparse.save_npz(self.path + name + 'gt_dominant.npz', data_d)
        sparse.save_npz(self.path + name + 'gt_recessive.npz', data_s)
        np.save(self.path + name + 'variants', variants_done)
        np.save(self.path + name + 'cadd', cadd_done)
        print('Total time in minutes: ', round((time.time() - start0) / 60, 2))
        return data_d, data_s, variants_done, cadd_done


    def fixing_erros(self, tag=''):
        """Fixing Errors"""
        data_df = sparse.load_npz(self.path + tag + 'gt_dominant.npz')
        # data_sf = sparse.load_npz(self.path  + tag + 'gt_recessive.npz')
        variants_f = np.load(self.path + tag + 'variants.npy')
        # cadd_f = np.load(self.path + tag + 'cadd.npy')
        samples = np.load(self.path + tag + 'samples.npy')

        missing = variants_f[data_df.shape[0]:len(variants_f)]
        row_ = G.sel(sample=samples, variant=missing).values.transpose()
        row_d = np.where(row_, 0, 1)
        # data_df = vstack([data_df, coo_matrix(row_d)])
        # row_ = G.sel(sample=samples, variant=variants_f).values.tranpose()
        # row_d = np.equal(row_, 0)
        # row_d = row_d * 1
        sparse.save_npz(path_output + tag + 'gt_dominant.npz', coo_matrix(row_d))


    def adding_known_snps_back(self, G, samples, variants_, cadd_,cadd_sort, data_df, data_sf):
        known_snps = pd.read_csv(self.path + 'known_snps_fullname.txt', header=None)[0].values

        """Adding known SNPS back"""
        for ks in known_snps:
            if ks not in variants_f:
                # print(ks)
                row_ = G.sel(sample=samples, variant=ks).values.transpose()
                row_d = np.where(row_, 0, 1)
                row_s = np.where(row_, 2, 1)

                data_df = vstack([data_df, coo_matrix(row_d)])
                data_sf = vstack([data_sf, coo_matrix(row_s)])

                variants_.append(ks)
                cadd_.append(cadd_sort.CADD_PHRED[cadd_sort.variants == ks].values[0])

        name = ''
        sparse.save_npz(self.path + name + 'gt_dominant.npz', data_df)
        sparse.save_npz(self.path + name + 'gt_recessive.npz', data_sf)
        np.save(self.path + name + 'variants', variants_)
        np.save(self.path + name + 'cadd', cadd_)


def eliminate_low_incidence(data, variants_name, cadd_score, ks_f):
    """
    Method used inside goPDX class
    :param data: snps file
    :param variants_name: snps names
    :param cadd_score:
    :param ks_f: known snps bool file
    :return: data, variants_name, cadd_score, ks_f filtered
    """
    # Remove columns whose less than 1% of counts are 1 in the data_df
    # Remove columns whose less than 1% of counts are 0 in the data_sf
    print('\n\nFiltering Low incidence')
    sum_df = np.array(data.sum(axis=0))
    th = np.ceil(data.shape[0] * 0.05)
    sum_df_greater = np.greater_equal(sum_df, th)  # SNPS less then th are False
    print('Before adding Known SNPS: ', sum_df_greater.sum())
    sum_df_greater = np.add(sum_df_greater[0, :], ks_f)
    print('After adding Known SNPS: ', sum(sum_df_greater))

    remove = []
    for i in range(len(sum_df_greater)):
        if not sum_df_greater[i]:
            remove.append(i)

    print('Original Shape Before Filtering: Data', data.shape, ', variants ', variants_name.shape,
          ', Cadd', cadd_score.shape, ', Ks', ks_f.shape)
    data = np.delete(data, remove, axis=1)
    variants_name = np.delete(variants_name, remove, axis=0)
    cadd_score = np.delete(cadd_score, remove, axis=0)
    ks_f = np.delete(ks_f, remove, axis=0)
    print('New Shape After Filtering: Data', data_df_.shape, ', variants ', variants_name.shape,
          ', Cadd', cadd_score.shape, ', Ks', ks_f.shape)
    return data, variants_name, cadd_score, ks_f


def eliminate_low_cadd(data, variants_name, cadd_score, ks_f):
    """
    Method used inside goPDX class
    :param data: snps file
    :param variants_name: snps names
    :param cadd_score: cadd score
    :param ks_f: known snps book file
    :return:data, variants_name, cadd_score, ks_f filtered
    """
    # Remove columns whose CADD less than 10
    print('\n\nFiltering Low Cadd')
    sum_df_greater = np.greater_equal(cadd_score, 10)  # SNPS less then th are False
    print('Before adding Known SNPS: ', sum_df_greater.sum())
    sum_df_greater = np.add(sum_df_greater, ks_f)
    print('After adding Known SNPS: ', sum(sum_df_greater))

    remove = []
    for i in range(len(sum_df_greater)):
        if not sum_df_greater[i]:
            remove.append(i)

    #data_df_ = data_df  # .todense().transpose()
    print('Original Shape Before Filtering: Data', data.shape, ', variants ', variants_name.shape,
          ', Cadd', cadd_score.shape, ', Ks', ks_f.shape)
    data = np.delete(data, remove, axis=1)
    variants_name = np.delete(variants_name, remove, axis=0)
    cadd_score = np.delete(cadd_score, remove, axis=0)
    ks_f = np.delete(ks_f, remove, axis=0)
    print('New Shape After Filtering: Data', data.shape, ', variants ', variants_name.shape,
          ', Cadd', cadd_score.shape, ', Ks', ks_f.shape)
    return data, variants_name, cadd_score, ks_f


def ld_prune(gn, variants, cadd, thold):
    """
    Method used inside goPDX class (filteringSNPS)
    input:
        subset of the gn, variants and cadd associated to this subset
    output:
        subset of the input subset without high correlated snps and cadd above 1

    """
    # https://en.wikipedia.org/wiki/Linkage_disequilibrium
    # Estimate the linkage disequilibrium parameter r for each pair of variants
    r = allel.rogers_huff_r(gn)
    correlations = squareform(r ** 2)
    correlations = pd.DataFrame(correlations)
    correlations.fillna(1, inplace=True)
    correlations = correlations.values
    del r
    # Saving the indiced of explored snps
    keep = []
    done = []

    for v_ in range(len(variants)):
        if v_ not in done:
            # Filtering out explored columns
            nextcolumns = set(np.arange(len(variants))) - set(done)
            filter_0 = np.zeros(len(variants))
            filter_0[list(nextcolumns)] = 1

            # Filtering the columns with high correlation
            filter_1 = np.greater(correlations[:, v_], thold)
            filter_1 = filter_1 * np.equal(filter_0, 1)

            if filter_1.sum() > 1:
                v_ind = np.arange(len(variants))[filter_1]
                v_ind = np.append(v_ind, v_)

                v_cadd = cadd[filter_1]
                v_cadd = np.append(v_cadd, cadd[v_])

                # keeping only the snp with highest cadd
                # if all less than 1, keep none
                filter_2 = np.equal(v_cadd, v_cadd.max())
                if v_cadd.max() > 1:
                    if isinstance(v_ind[filter_2], np.ndarray):
                        keep.append(v_ind[filter_2][0])
                    else:
                        keep.append(v_ind[filter_2])

                for item in v_ind:
                    done.append(item)
            else:
                keep.append(v_)
                done.append(v_)

    # Filtering final results on the subset to output
    # ADD FUNCTION TO KEEP KNOWN ELEMENTS HERE
    loc_unlinked = np.zeros(len(variants))
    loc_unlinked[keep] = 1

    gn = gn.compress(loc_unlinked, axis=0)
    variants = variants[keep]
    cadd = cadd[keep]
    return gn, variants, cadd



