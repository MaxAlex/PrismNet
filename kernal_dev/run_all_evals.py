import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as pyt
from collections import defaultdict
import sqlite3
import re
import multiprocessing
from functools import reduce
import numpy as np
from multiplierz.fasta import parse_to_generator

import prismnet.functional_main as prism
import prismnet.convert_dataset as convert


model_files = glob('/data/workspace/deep_rna/prismnet/PrismNet/exp/prismnet/out/models/*')
nonstructure_models = glob('/data/workspace/deep_rna/prismnet/nonstructure_models/models/*')
tsv_input = glob('/data/workspace/deep_rna/prismnet/PrismNet/data/clip_data/*tsv')
h5_input = glob('/data/workspace/deep_rna/prismnet/PrismNet/data/clip_data/*h5')

p_model = {os.path.basename(x).split('_')[0]: x for x in model_files}
p_tsv = {os.path.basename(x).split('_')[0]: x for x in tsv_input}
p_h5 = {os.path.basename(x).split('_')[0]: x for x in h5_input}

fiveutr_sequences = [(h, s) for h, s in parse_to_generator('/data/workspace/fastas/biomart_5UTR_sequences.fasta')]
threeutr_sequences = [(h, s) for s, h in pd.read_csv('/data/workspace/fastas/biomart_3UTR_sequences_fixed.tsv', sep='\t').values]

default_struct_str = ','.join(['-1'] * 101)


def evaluate_sequence_list(sequences, model_file, temp_dir, type_str='A', score_str='1.0',
                           struct_str=default_struct_str):
    if not os.path.exists(os.path.join(temp_dir, 'out')):
        os.mkdir(os.path.join(temp_dir, 'out'))

    sequence_rows = []
    for name, seq in sequences:
        row = '\t'.join([type_str, name, seq, struct_str, score_str, '1'])
        sequence_rows.append(row)

    tsv_file = os.path.join(temp_dir, 'eval_temp.tsv')
    with open(tsv_file, 'w') as out:
        out.write('\n'.join(sequence_rows) + '\n')
    print(tsv_file)
    h5_file = convert.convert(tsv_file)

    eval_file = prism.run_infer(tsv_file, model_file, 'foobar', temp_dir, mode='seq', use_structure=False,
                                cuda_mode=False)
    evals = []
    with open(eval_file, 'r') as inp:
        for line in inp:
            if not line.strip():
                break
            #             prob, label = .strip().split()
            evals.append(float(line.strip()))
    assert (len(evals) == len(sequences)), (eval_file, len(evals), len(sequences))

    return [(name, ev) for (name, seq), ev in zip(sequences, evals)]


# "jump_size" is how offset each subset is from each other; so, 50 so that each subseq overlaps ~half with its neighbors
def evaluate_transcripts(transcripts, *etc, start=None, stop=None, seq_len=101, jump_size=50, eval_count_size=100000,
                         **etcetc):
    all_subseqs = []
    for name, transcript in transcripts:
        if len(transcript) < 101:
            continue
        assert ('___' not in name)
        subtran = transcript[start:stop]
        for i in range(0, len(transcript), jump_size):
            subseq = subtran[i:i + seq_len]
            if len(subseq) != seq_len:
                assert (i + seq_len > len(subtran)), (len(subtran), i, seq_len, jump_size)
                subseq = subtran[-seq_len:]
                assert (len(subseq) == seq_len), (len(subtran), i, seq_len, len(transcript))
            all_subseqs.append((name + '___%s' % i, subseq))

    results = []
    for i in range(0, len(all_subseqs), eval_count_size):
        print("B_%d" % i)
        subseqs = all_subseqs[i:i + eval_count_size]

        evaled_subseqs = evaluate_sequence_list(subseqs, *etc, **etcetc)

        for tran, subs in collectByCriterion(evaled_subseqs, lambda x: x[0].split('___')[0]).items():
            scores = [x[1] for x in subs]
            results.append((tran, np.mean(scores), np.median(scores), np.max(scores)))

    return results


backup_dir = '/data/workspace/temp/model_eval_backups'

def run_model_on_stuff(model):
    backup_file = os.path.join(backup_dir, os.path.basename(model).split('.')[0] + '_evals.csv')
    if os.path.exists(backup_file):
        tab = pd.read_csv(backup_file).set_index('Name')
        evals.append(tab)
    else:
        rbp = os.path.basename(model).split('.')[0]
        five_scores = evaluate_transcripts(fiveutr_sequences,
                                           model,
                                           '/data/workspace/temp')
        five_tab = pd.DataFrame(five_scores,
                                columns=['Name', rbp + "_5_Mean", rbp + "_5_Median", rbp + "_5_MaxVal"]
                                ).set_index('Name')
        three_scores = evaluate_transcripts(threeutr_sequences,
                                            model,
                                            '/data/workspace/temp')
        three_tab = pd.DataFrame(three_scores,
                                 columns=['Name', rbp + "_3_Mean", rbp + "_3_Median", rbp + "_3_MaxVal"]
                                 ).set_index('Name')
        scores = pd.merge(five_tab, three_tab, left_index=True, right_index=True, how='outer')
        scores.to_csv(backup_file)

    return scores


if __name__ == '__main__':
    evals = multiprocessing.Pool().map(run_model_on_stuff, nonstructure_models)

    full_table = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), evals)
    full_table.to_csv('/data/workspace/deep_rna/prismnet/full_nonstructure_model_evals.csv')

    print("Done.")