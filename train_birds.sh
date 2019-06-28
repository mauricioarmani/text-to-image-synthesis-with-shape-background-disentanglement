# name='model1' # SEG 10 - BG 10
# name='model2' # SEG 10 - BG 20
# name='model3' # SEG 0 - BG 10
# name='model4' # SEG 10 - BG 50
# name='model5' # manipulate = True
# name='model6' # manipulate = False - L1Loss
# name='model7' # manipulate = False - L1Loss - "sem shape"
name='test'
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
                                # --manipulate
                                # --reuse_weights \
                                # --load_from_epoch 440