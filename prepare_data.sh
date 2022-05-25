src_l="en"
tgt_l="vi"

rm -f /home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/*.${src_l}-${tgt_l}.*
!python3 /home/tqkhuong/Documents/THESIS/preprocessIWSLT15/prepare_corpus.py ./data15 /home/tqkhuong/Documents/THESIS/preprocessIWSLT15/preprocess_data -s $src_l -t $tgt_l 