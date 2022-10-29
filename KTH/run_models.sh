cd ..
# python3 main.py --EPOCHS 20 --MODEL NCF --NCF_DROPOUT 0.4 --LR 1e-3 --NCF_EMBED_DIM 32
# python3 main.py --EPOCHS 10 --FM_EMBED_DIM 8 --MODEL FM
# python main.py --MODEL DCN
# python main.py --MODEL WDN
python main.py --MODEL CNN_FM --BATCH_SIZE 256
# python main.py --MODEL DeepCoNN
# python main.py --MODEL FFM
