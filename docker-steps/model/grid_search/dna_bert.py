import os

if __name__ == '__main__':
    for rnn_layer in ['lstm', 'gru']:
        for n_attention_heads in [1, 2, 4, 8, 12]:
            for hidden in [1024]:
                for n_hidden_layer in [7, 9, 12]:
                    for n_rnn_layers in [1, 2, 3, 4]:
                        for n_words in [20, 30]:
                            command: str = f'python3 {os.path.join(os.getcwd(), "train_gene_classifier.py ")}' \
                                           f'-len_read 150 ' \
                                           f'-len_kmer 6 ' \
                                           f'-n_words {n_words} ' \
                                           f'-batch 512 ' \
                                           f'-model_selected dna_bert ' \
                                           f'-tokenizer_selected dna_bert_n ' \
                                           f'-hidden_size {hidden} ' \
                                           f'-dropout 0.6 ' \
                                           f'-n_attention_heads {n_attention_heads} ' \
                                           f'-n_beams 1 ' \
                                           f'-n_hidden_layers {n_hidden_layer} ' \
                                           f'-rnn {rnn_layer} ' \
                                           f'-n_rnn_layers {n_rnn_layers} ' \
                                           f'-grid_search True'
                            # train model
                            os.system(command)
