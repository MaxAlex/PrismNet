import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as pyt
from collections import defaultdict
import sqlite3
import re
import multiprocessing

from Bio import pairwise2

from random import seed, shuffle
seed(0)


tsv_input = glob('/data/workspace/deep_rna/prismnet/PrismNet/data/clip_data/*tsv')
infer_files = glob('/data/workspace/deep_rna/prismnet/test_inferences/out/infer/*')
p_tsv = {os.path.basename(x).split('_')[0]: x for x in tsv_input}
p_infer = {os.path.basename(x).split('_')[4]: x for x in infer_files}


def load_with_infer(prot):
    tab = pd.read_csv(p_tsv[prot], sep='\t',
                      names=['Type', 'Name', 'Seq', 'icshape', 'Score', 'label'])
    tab = tab[tab.Type != 'Type']  ## WHYYYYY DONT THEY STANDARDIZE THEIR DATA???????
    infer = [float(x) for x in open(p_infer[prot]).read().strip().split('\n')]
    assert (len(infer) == tab.shape[0])
    tab['Infer'] = infer

    return tab


def align_callback(a1, a2):
    if a1 == 'N' or a2 == 'N':
        return 1
    elif a1 == a2:
        return 1
    else:
        return -0




def shuffle_string(thing):
    foo = list(thing)
    shuffle(foo)
    return ''.join(foo)

def align_to_sequence(motif, sequence):
    aln = pairwise2.align.localcd(motif, sequence, align_callback, -999, -999, -999, -999)
    scores = [x.score for x in aln]
    # print(aln)
    return max(scores) # / min(len(motif), len(sequence))


def generate_scores_table(prot):
    outputfile = '/data/workspace/deep_rna/prismnet/test_inferences/motif_match_checks/%s_table.csv'

    print(prot)
    tab = load_with_infer(prot)
    tab.to_csv(outputfile)
    for m_index, motif in enumerate(prot_motifs[prot]):
        tab['M%d_scores' % m_index] = tab.Seq.apply(lambda x: align_to_sequence(motif, x))

        cont_motif = shuffle_string(motif)
        tab['M%d_random_control' % m_index] = tab.Seq.apply(lambda x: align_to_sequence(cont_motif, x))

        print('\t%s %s' % (motif, cont_motif))

    tab.to_csv(outputfile)


if __name__ == '__main__':
    # In order to get this to work, had to load the provided sql dump into mysql, re-dump it out
    # with --skip-extended-insert and --compact options set, and then use https://github.com/dumblob/mysql2sqlite
    # to convert to a sqlite3 DB

    conn = sqlite3.connect('/data/workspace/RBP/rbpdb_files/RBPDB.db')
    rbp_prots = pd.read_sql_query('SELECT * FROM proteins', conn)
    rbp_exps = pd.read_sql_query('SELECT * FROM experiments', conn)
    rbp_protexps = pd.read_sql_query('SELECT * FROM protExp', conn)

    rbp_exps = rbp_exps[rbp_exps.flag != 1.0]
    rbp_prots = rbp_prots[rbp_prots.species == 'Homo sapiens']

    prot_from_pid = dict(zip(rbp_prots['id'], rbp_prots['geneName']))
    pid_from_eid = dict(zip(rbp_protexps['expID'], rbp_protexps['protID']))  # Incorrect annotation!

    rbp_exps['Protein'] = rbp_exps['id'].apply(lambda x: prot_from_pid.get(pid_from_eid.get(x, 'NA_1'), 'NA_2'))

    prot_motifs = defaultdict(set)
    for i, row in rbp_exps.iterrows():
        if (row['Protein'] != 'NA_1' and row['Protein'] != 'NA_2' and
                row['sequence_motif'] and len(row['sequence_motif']) > 5 and
                set(row['sequence_motif']).issubset({'A', 'C', 'U', 'G', 'N'})):
            prot_motifs[row['Protein']].add(row['sequence_motif'])

    prismnet_prots = {os.path.basename(x).split('_')[0] for x in
                      glob('/data/workspace/deep_rna/prismnet/PrismNet/exp/prismnet/out/models/*')}
    common_prots = prismnet_prots & set(prot_motifs.keys())

    multiprocessing.Pool(8).map(generate_scores_table, common_prots)
    print("Done.")