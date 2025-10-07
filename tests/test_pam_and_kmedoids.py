"""
@Author  : 李欣怡
@File    : test_pam_and_kmedoids.py
@Time    : 2025/6/16 16:57
@Desc    : test PAM and Kmedoids from Pull Request of Deng Cheng
"""
import numpy as np

from sequenzo import *
from sequenzo.clustering import KMedoids

df = load_dataset('country_co2_emissions')

time_list = list(df.columns)[1:]
states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
sequence_data = SequenceData(df,
                             time=time_list,
                             id_col="country",
                             states=states,
                             labels=states)

om = get_distance_matrix(seqdata=sequence_data,
                         method='OMspell',
                         sm="TRATE",
                         indel="auto")
print("Distance matrix:\n", om)

print(" - test PAM ...")
pam = KMedoids(method="PAM",
              k=5,
              initialclust=[1, 2, 3, 4, 5],
              npass=5,
              diss=om)
print('clustering result:\n', pam)

print(" - test KMedoids ...")
kmedoids = KMedoids(method="KMedoids",
                    k=5,
                    initialclust=[1, 2, 3, 4, 5],
                    npass=5,
                    diss=om)
print('clustering result:\n', kmedoids)

print(" - test PAMoce ...")
pamonce = KMedoids(method="PAMonce",
                    k=5,
                    initialclust=[1, 2, 3, 4, 5],
                    npass=5,
                    diss=om)
print('clustering result:\n', pamonce)