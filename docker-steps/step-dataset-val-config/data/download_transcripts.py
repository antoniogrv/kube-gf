from typing import Final
from typing import Dict
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
import requests
import sys
import os

"""
1 - Rest API Ensemble.org https://rest.ensembl.org
2 - GET xrefs/symbol/:species/:symbol https://rest.ensembl.org/documentation/info/xref_external
3 - GET sequence/id/:id https://rest.ensembl.org/documentation/info/sequence_id
"""

SERVER: Final = "https://rest.ensembl.org"


def get_ids(
        gene_value: str
) -> List[str]:
    # init query
    ext: str = f'/xrefs/symbol/homo_sapiens/{gene_value}?object_type=gene'
    # get result
    r = requests.get(
        SERVER + ext,
        headers={
            'Content-Type': 'application/json'
        })
    # if response status is not okay
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    # init list of results
    id_list: List[str] = []
    data_response = r.json()
    # add each result in id_list array
    for data in data_response:
        id_list.append(data['id'])

    return id_list


def get_sequences(
        id_sequence_value: str
) -> str:
    # init query
    ext: str = f'/sequence/id/{id_sequence_value}?type=cdna;multiple_sequences=1'
    # get result
    r = requests.get(
        SERVER + ext,
        headers={
            'Content-Type': 'text/x-fasta'
        })
    # if response status is not okay
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    # return results
    return r.text


def create_file(
        root_dir: str,
        gene_value: str,
        sequence_value: str
) -> None:
    with open(os.path.join(root_dir, f'{gene_value}.fastq'), 'a') as fasta_file:
        fasta_file.write(sequence_value)


if __name__ == '__main__':
    # init gene dictionary
    gene_dict: Dict[str, List[str]] = {}
    # init paths
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))
    genes_panel_path: str = os.path.join(
        os.getcwd(),
        os.getenv('GENES_PANEL_LOCAL_PATH')
    )
    transcript_dir_path: str = os.path.join(
        os.getcwd(),
        os.getenv('TRANSCRIPT_LOCAL_DIR')
    )

    # check if transcripts dir exists
    if not os.path.exists(transcript_dir_path):
        os.makedirs(transcript_dir_path)

    # for each gene in genes_panel_file
    with open(genes_panel_path, 'r') as genes_panel_file:
        for gene in tqdm(genes_panel_file, desc='Downloading the transcripts of the genes...'):
            # remove new line character
            gene: str = gene.rstrip('\n')
            gene_dict[gene] = get_ids(gene)
            for sequence_id in gene_dict[gene]:
                sequence = get_sequences(sequence_id)
                create_file(transcript_dir_path, gene, sequence)
