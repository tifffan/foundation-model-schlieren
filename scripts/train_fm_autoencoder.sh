python src/models/autoencoders/fm_features/main.py \
    --learning_rate 1e-3 \
    --data_path data/features_16/exp_frames_672_432/dino_features_16.npy \
    --dataset_keyword dino_features_16 \
    --model_keyword unet_ae_5blk \
    --latent_dim 64 \
    --epochs 100 \
    --batch_size 32 \
    --conv_channels 8,8,8,16,16 \
    --random_seed 42 \
    --subsample_size 256
    