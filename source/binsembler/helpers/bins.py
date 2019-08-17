import pandas as pd

# This file contains some helper functions to specifically use in this project.
# Bins file contains common functions to create bins

def get_bins(probs, bins=10):
    """
    Splits the probs list into bins of size nbins and calculates the total counts of probs in each bin
    :param probs:
    :param nbins:
    :return: a pandas datframe with bin label and number of records in the bin
    """
    prob = [x * 100 for x in probs]
    bin_df = pd.DataFrame()
    bins_prob = [i for i in range(0, 101, bins)]
    names_prob = [str(i) + str('-') + str(i + bins) for i in range(0, 100, bins)]
    bin_df['bins'] = pd.cut(prob, bins_prob, labels=names_prob)
    bin_df['bins'] = bin_df['bins'].astype('str')
    bin_df['prob'] = prob
    temp = bin_df.groupby(['bins']).agg({'prob': 'count'}).rename(columns={'prob': 'counts'}).reset_index()
    bin_df = pd.merge(bin_df, temp, how='left', on=['bins'])
    return bin_df


def get_bins_full(probs, nbins=10):
    """
    Splits the probs list into bins of size nbins and calculates the total counts of probs in each bin
    :param probs:
    :param nbins:
    :return: a pandas datframe with bin label and number of records in the bin
    """
    prob = [x * 100 for x in probs]
    bin_df = pd.DataFrame()
    bins_prob = [i for i in range(0, 101, nbins)]
    # names_prob = [(str(i) + str('-') + str(i + nbins), i, i + nbins) for i in range(0, 100, nbins)]
    names_prob = [str(i) + str('-') + str(i + nbins) for i in range(0, 100, nbins)]
    bin_df['bins'] = pd.cut(prob, bins_prob, labels=names_prob)
    bin_df['bins'] = bin_df['bins'].astype('str')
    bin_df['prob'] = prob
    temp = bin_df.groupby(['bins']).agg({'prob': 'count'}).rename(columns={'prob': 'counts'}).reset_index()
    temp1 = pd.DataFrame(names_prob, columns=['bins'])
    temp1 = pd.merge(temp1, temp, how='left', on=['bins']).fillna(0)
    return temp1
