# DeepFM
# python main.py --MODEL DFM --PATIENCE_LIMIT 100 --LR 1e-4 --DFM_EMBED_DIM 128 --DFM_HIDDEN_UNITS 4 --DFM_DROPOUT 0.2 --DFM_USE_BN True --EPOCHS 100 --MESSAGE embed_dim_5_dropout_0.2_use_bn_True_epochs_10_state_language_category
# python main.py --MODEL DFM --PATIENCE_LIMIT 100 --LR 1e-4 --DFM_EMBED_DIM 32 --DFM_HIDDEN_UNITS 512 256 --DFM_DROPOUT 0.2 --DFM_USE_BN True --EPOCHS 10 --MESSAGE embed_dim_5_dropout_0.2_use_bn_True_epochs_10_state_language_category
# python main.py --MODEL DFM --PATIENCE_LIMIT 100 --LR 1e-4 --DFM_EMBED_DIM 32 --DFM_HIDDEN_UNITS 512 --DFM_DROPOUT 0.2 --DFM_USE_BN True --EPOCHS 2 --MESSAGE embed_dim_32_dropout_0.2_use_bn_True_epochs_2_state_category_new_language_year

# python main.py --MODEL FM --EPOCHS 20 --FM_EMBED_DIM 4 --MESSAGE embed_dim_4_epochs_20_preprocessed_state_language_category_year

python main.py --MODEL DCN --EPOCHS 10 --PATIENCE_LIMIT 100

# python ensemble.py --ENSEMBLE_FILES 20221029_054718_DFM_embed_dim_5_dropout_0.2_use_bn_True_epochs_1,20221101_142635_DFM_embed_dim_32_dropout_0.2_use_bn_True_epochs_2_state_category_new_language_year --ENSEMBLE_WEIGHT 0.6,0.4

# python main_jhk.py --MODEL NCF --EPOCHS 100 --ADD_CONTEXT isbn
