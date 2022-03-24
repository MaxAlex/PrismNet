import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as pyt
from collections import defaultdict
import sqlite3
import re
import multiprocessing

from prismnet.functional_main import *

model_files = glob('/data/workspace/deep_rna/prismnet/PrismNet/exp/prismnet/out/models/*')
tsv_input = glob('/data/workspace/deep_rna/prismnet/PrismNet/data/clip_data/*tsv')
h5_input = glob('/data/workspace/deep_rna/prismnet/PrismNet/data/clip_data/*h5')

p_model = {os.path.basename(x).split('_')[0]: x for x in model_files}
p_tsv = {os.path.basename(x).split('_')[0]: x for x in tsv_input}
p_h5 = {os.path.basename(x).split('_')[0]: x for x in h5_input}

def run_infer_instance(prot):
    run_infer(p_tsv[prot], p_model[prot], 'March21_run',
              '/data/workspace/deep_rna/prismnet/test_inferences', cuda_mode=False,
              workers=0)

if __name__ == '__main__':
    # Skipping these, already done
    rbpdb_prots = {'TRA2A', 'TIAL1', 'HNRNPD', 'FUS', 'QKI', 'LSM11', 'HNRNPF', 'TARDBP', 'IGF2BP1', 'PCBP1',
                   'CSTF2T', 'CSTF2', 'AKAP1', 'HNRNPA1', 'LIN28B', 'ELAVL1', 'NONO', 'U2AF2', 'TIA1', 'PCBP2',
                   'HNRNPK', 'SLBP', 'PUM1', 'FMR1', 'SLTM', 'KHDRBS1', 'HNRNPC', 'KHSRP', 'PTBP1', 'PUM2'}

    all_prots = {os.path.basename(x).split('_')[0] for x in model_files}

    multiprocessing.Pool(8).map(run_infer_instance, (all_prots - rbpdb_prots))
    #list(map(run_infer_instance, (all_prots - rbpdb_prots)))
    print("Done.")

