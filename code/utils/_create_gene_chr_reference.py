# Create a reference file of genes and corresponding chromosomes
# Save the file in ./data

# Example call:
# python _create_gene_chr_reference.py /data100t1/home/wanying/shared_data_files/Ensembl/Homo_sapiens.GRCh38.112.gtf.gz

import pandas as pd
import numpy as np
import sys


def get_gene_id(val):
    '''
    Get gene ensmbl id from the last column of a gtf file
    '''
    lst = val.split('; ')
    for v in lst:
        if 'gene_id' in v:
            return v.split(' ')[-1].strip('"')
    return np.nan

if __name__ == '__main__':
    ref_fn = sys.argv[1]
    print(f'# Create reference file from: {ref_fn}')

    df = pd.read_csv(ref_fn, sep='\t', comment='#', header=None, dtype=str, compression='gzip')
    df_cleaned = df[df[2]=='gene'].copy()

    print('\n# Available chromosomes in the file')
    print(df_cleaned[0].unique())

    print('\n# Only keep genes on autosomes')
    df_cleaned = df_cleaned[df_cleaned[0].isin([str(x) for x in range(1, 23)])]
    df_cleaned['gene_id'] = df_cleaned.iloc[:, -1].apply(get_gene_id)
    df_cleaned.rename(columns={0:'chr'}, inplace=True)

    cols_to_save = ['gene_id', 'chr']
    df_cleaned = df_cleaned[cols_to_save].dropna()

    output_fn = './data/gene_chr_reference.txt'
    print(f'\n# Save all genes to output file {output_fn}')
    df_cleaned.to_csv(output_fn, sep='\t', index=False)


    print(f'\n# Save genes by chromosome to output files')
    for chr_num, df_single_chr in df_cleaned.groupby('chr'):
        print(f'# - chr{chr_num}')
        output_fn = f'./data/gene_chr_reference.chr{chr_num}.txt'
        df_single_chr.to_csv(output_fn, sep='\t', index=False)

    print('\n# Done')