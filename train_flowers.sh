name='test'
device=3
dataset='flowers'
CUDA_VISIBLE_DEVICES=${device} python gan/train_worker.py \
                                --dataset $dataset \
                                --batch_size 32 \
                                --model_name ${name} \
                                --g_lr 0.0002 \
                                --d_lr 0.0002 \
                                --save_freq 10 \
                                --unet_checkpoint 'segmentation/checkpoints/checkpoint590.pt' \
                                # --manipulate
                                # --reuse_weights \
                                # --load_from_epoch 440
