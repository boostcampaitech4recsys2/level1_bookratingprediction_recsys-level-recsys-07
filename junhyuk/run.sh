# python3 main.py --EPOCHS 20 --FM_EMBED_DIM 8 --MODEL FM
# python3 main.py --MODEL FFM
# python3 main.py --MODEL NCF
# python3 main.py --EPOCHS 1 --MODEL NCF
# python3 main.py --EPOCHS 6 --MODEL WDN
# python3 main.py --EPOCHS 20 --MODEL DCN
# python3 main.py --MODEL CNN_FM

# DeepCoNN: user_summary_merge_vector, item_summary_vector -> 생성해야하면
# 1:19~
# python3 main.py --EPOCHS 20 --MODEL DeepCoNN --DEEPCONN_VECTOR_CREATE True
# python3 main.py --EPOCHS 20 --MODEL DeepCoNN #model
# python ensemble.py --ENSEMBLE_FILES 20221027_012536_NCF,20221027_020557_FM,20221027_145216_WDN,20221027_145249_DCN --ENSEMBLE_STRATEGY WEIGHTED --ENSEMBLE_WEIGHT 0.40,0.35,0.15,0.1

# python3 main.py --EPOCHS 6 --MODEL WDN
# python3 main.py --EPOCHS 4 --MODEL DCN
# python3 main.py --EPOCHS 4 --MODEL FFM
# def summary_merge(df, user_id, max_summary):
#     return " ".join(df[df['user_id'] == user_id].sort_values(by='summary_length', ascending=False)['summary'].values[:max_summary])

# python ensemble.py --ENSEMBLE_FILES 20221027_012536_NCF,DCN_2.1816_msk,20221101_145823_WDN --ENSEMBLE_STRATEGY WEIGHTED --ENSEMBLE_WEIGHT 0.40,0.5,0.1
python ensemble.py --ENSEMBLE_FILES ensemble_2.1643,DCN_msk --ENSEMBLE_STRATEGY WEIGHTED --ENSEMBLE_WEIGHT 0.8,0.2
