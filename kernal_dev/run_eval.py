import os

import prismnet.functional_main as prism
import prismnet.convert_dataset as convert

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

    eval_file = prism.run_eval(h5_file, model_file, temp_dir, mode='seq', cuda_mode=False)
    evals = []
    with open(eval_file, 'r') as inp:
        for line in inp:
            if not line.strip():
                break
            prob, label = line.strip().split()
            evals.append(float(prob))
    assert (len(evals) == len(sequences))

    return [(name, seq, ev) for (name, seq), ev in zip(sequences, evals)]


if __name__ == '__main__':
    foobar = [('Aha', 'A' * 101), ('Cat', 'C' * 101), ('Gob', 'G' * 101), ('Tap', 'T' * 101)]
    evaluate_sequence_list(foobar,
                           '/mnt/c/Users/WilliamMaxAlexander/Projects/deep_rna/PrismNet/nonstructure_models/RBPMS_HEK293.h5_PrismNet_seq_best.pth',
                           '/mnt/c/Users/WilliamMaxAlexander/Projects/deep_rna')