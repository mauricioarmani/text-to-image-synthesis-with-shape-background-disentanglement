# env = hdgan2
name='SEG10_BG50L2_birds'
# name='SEG10_BG10L1_IDTLOSS_birds'
epoch='500'
device=2

# ALL ALIGNED
# align='all'
# h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
# CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
#             --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder results \
#             --h5_file ${h5_file} \
#             --batch_size 10 \
#             --num_classes 50
# CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
#         --image_folder results \
#         --h5_file ${h5_file}            

# ALL ALIGNED AND RANDOM SHAPE NOISE
# align='all'
# h5_file="$name""_G_epoch_""$epoch""_align_""$align""_shape_noise"".h5"
# CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
#             --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder results \
#             --h5_file ${h5_file} \
#             --batch_size 10 \
#             --num_classes 50
# CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
#         --image_folder results \
#         --h5_file ${h5_file} 

# ALL ALIGNED AND RANDOM BACKGROUND NOISE
align='all'
h5_file="$name""_G_epoch_""$epoch""_align_""$align""_background_noise"".h5"
CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
            --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50
CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
        --image_folder results \
        --h5_file ${h5_file} 

# ALL ALIGNED AND RANDOM BACKGROUND AND SHAPE NOISES
align='all'
h5_file="$name""_G_epoch_""$epoch""_align_""$align""_shape_noise_background_noise"".h5"
CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
            --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50
CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
        --image_folder results \
        --h5_file ${h5_file} 

# # SHAPE ALIGNED
# align='shape'
# h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
# CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
#             --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder results \
#             --h5_file ${h5_file} \
#             --batch_size 10 \
#             --num_classes 50
# CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
#         --image_folder results \
#         --h5_file ${h5_file}

# # BACKGROUND ALIGNED
# align='background'
# h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
# CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
#             --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder results \
#             --h5_file ${h5_file} \
#             --batch_size 10 \
#             --num_classes 50
# CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
#         --image_folder results \
#         --h5_file ${h5_file}

# # NOTHING ALIGNED
# align='none'
# h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
# CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
#             --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder results \
#             --h5_file ${h5_file} \
#             --batch_size 10 \
#             --num_classes 50            
# CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
#         --image_folder results \
#         --h5_file ${h5_file}