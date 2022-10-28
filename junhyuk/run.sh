# python3 main.py --EPOCHS 20 --FM_EMBED_DIM 8 --MODEL FM
# python3 main.py --MODEL FFM
# python3 main.py --MODEL NCF
python3 main.py --EPOCHS 20 --MODEL NCF --NCF_DROPOUT 0.1 --BATCH_SIZE 1024 --LR 1e-3
# python3 main.py --EPOCHS 6 --MODEL WDN
# python3 main.py --EPOCHS 20 --MODEL DCN
# python3 main.py --MODEL CNN_FM

# DeepCoNN: user_summary_merge_vector, item_summary_vector -> 생성해야하면
# python3 main.py --MODEL DeepCoNN --DEEPCONN_VECTOR_CREATE True
# python3 main.py --MODEL DeepCoNN #model 
# python ensemble.py --ENSEMBLE_FILES 20221027_012536_NCF,20221027_020557_FM,20221027_145216_WDN,20221027_145249_DCN --ENSEMBLE_STRATEGY WEIGHTED --ENSEMBLE_WEIGHT 0.40,0.35,0.15,0.1

# python3 main.py --EPOCHS 6 --MODEL WDN
# python3 main.py --EPOCHS 4 --MODEL DCN
# python3 main.py --EPOCHS 4 --MODEL FFM