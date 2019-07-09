# name='model1' # SEG 10 - BG 10
# name='model2' # SEG 10 - BG 20
# name='model3' # SEG 0 - BG 10
# name='model4' # SEG 10 - BG 50

# name='SEG10_BG10L1_IDTLOSS_OBJLOSS_birds'

name='SEG10_BG10L1_0KL'
device=3
dataset='birds'
CUDA_VISIBLE_DEVICES=${device} python gan/train_worker.py \
                                --dataset $dataset \
                                --batch_size 64 \
                                --model_name ${name} \
                                --g_lr 0.0002 \
                                --d_lr 0.0002 \
                                --save_freq 10 \
                                --unet_checkpoint 'segmentation/checkpoints/checkpoint590.pt' \
                                # --reuse_weights \
                                # --load_from_epoch 390
                                # --manipulate \                             