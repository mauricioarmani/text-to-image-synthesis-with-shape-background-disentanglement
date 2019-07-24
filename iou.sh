# env = pytorch
# name='SEG10_BG10L1_birds'
# name='SEG10_BG10L1_0KL_birds'
name='SEG10_0KL_birds'
epoch='500'
device=0

# # ALL ALIGNED
# align='all'
# CUDA_VISIBLE_DEVICES=${device} python evaluation/iou/iou.py --model_name ${name} \
#                                                             --unet_checkpoint 'segmentation/checkpoints/checkpoint590.pt'\
#                                                             --load_from_epoch ${epoch} \
#                                                             --align ${align}

# # ALL ALIGNED AND RANDOM BACKGROUND NOISE
# align='all'
# CUDA_VISIBLE_DEVICES=${device} python evaluation/iou/iou.py --model_name ${name} \
#                                                             --unet_checkpoint 'segmentation/checkpoints/checkpoint590.pt'\
#                                                             --load_from_epoch ${epoch} \
#                                                             --align ${align} \
#                                                             --background_noise 

# # SHAPE ALIGNED
# align='shape'
# CUDA_VISIBLE_DEVICES=${device} python evaluation/iou/iou.py --model_name ${name} \
#                                                             --unet_checkpoint 'segmentation/checkpoints/checkpoint590.pt'\
#                                                             --load_from_epoch ${epoch} \
#                                                             --align ${align}

# # BACKGROUND ALIGNED
# align='background'
# CUDA_VISIBLE_DEVICES=${device} python evaluation/iou/iou.py --model_name ${name} \
#                                                             --unet_checkpoint 'segmentation/checkpoints/checkpoint590.pt'\
#                                                             --load_from_epoch ${epoch} \
#                                                             --align ${align}

# # NOTHING ALIGNED
# align='none'
# CUDA_VISIBLE_DEVICES=${device} python evaluation/iou/iou.py --model_name ${name} \
#                                                             --unet_checkpoint 'segmentation/checkpoints/checkpoint590.pt'\
#                                                             --load_from_epoch ${epoch} \
#                                                             --align ${align}