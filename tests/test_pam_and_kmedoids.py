"""
@Author  : 李欣怡
@File    : test_pam_and_kmedoids.py
@Time    : 2025/6/16 16:57
@Desc    : test PAM and Kmedoids from Pull Request of Deng Cheng
"""
import numpy as np

from sequenzo import *
from sequenzo.clustering import PAM, KMedoids

df = load_dataset('country_co2_emissions')

time_list = list(df.columns)[1:]
states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
sequence_data = SequenceData(df,
                             time=time_list,
                             time_type="year",
                             id_col="country",
                             states=states,
                             labels=states)

om = get_distance_matrix(seqdata=sequence_data,
                         method='OMspell',
                         sm="TRATE",
                         indel="auto")

weight = np.ones(sequence_data.seqdata.shape[0], dtype=int)

print(" - test PAM ...")
pam = PAM(nelements=sequence_data.seqdata.shape[0],
          diss=om,
          centroids=[1, 2, 3, 4, 5],
          npass=5,
          )
clustering = pam.runclusterloop()

print(" - test KMedoids ...")
kmedoids = KMedoids(nelements=sequence_data.seqdata.shape[0],
                    diss=om,
                    centroids=[1, 2, 3, 4, 5],
                    npass=5,
                    )
clustering = kmedoids.runclusterloop()