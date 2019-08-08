# env = hdgan2
# iou env = pytorch
# name='BG10L1_birds'
# name='SEG10_birds'
# name='charCNNRNN_birds'
# name='SEG10_BG10L1_char_birds'
# name='SEG10_BG10L1_birds'
# name='SEG10_BG10L1_0KL_birds'
# name='SEG10_char_birds'
# name='BASELINE_birds'
# name='SEG10_0KL_birds'
epoch='500'
device=0

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
# CUDA_VISIBLE_DEVICES=${device} python evaluation/neudist/neudist.py \
#                         --dataset birds \
#                         --testing_path results/${h5_file} \
#                         --model_name neudist_birds \
#                         --load_from_epoch 595
# h5_file="$name""_G_epoch_""$epoch"".h5"
# python evaluation/ms_ssim/msssim_score.py \
        # --image_folder results \
        # --h5_file ${h5_file} \
        # --evaluate_overall_score \
        # --sample_per_cls 100


# # ALL ALIGNED AND RANDOM SHAPE NOISE
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

# # ALL ALIGNED AND RANDOM BACKGROUND NOISE
# align='all'
# h5_file="$name""_G_epoch_""$epoch""_align_""$align""_background_noise"".h5"
# CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
#             --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder results \
#             --h5_file ${h5_file} \
#             --batch_size 10 \
#             --num_classes 50
# CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
#         --image_folder results \
#         --h5_file ${h5_file} 
# python evaluation/ms_ssim/msssim_score.py \
#         --image_folder results \
#         --h5_file ${h5_file} \
#         --evaluate_overall_score \
#         --sample_per_cls 200

# # ALL ALIGNED AND RANDOM BACKGROUND AND SHAPE NOISES
# align='all'
# h5_file="$name""_G_epoch_""$epoch""_align_""$align""_shape_noise_background_noise"".h5"
# CUDA_VISIBLE_DEVICES=${device} python evaluation/inception_score/inception_score.py \
#             --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
#             --image_folder results \
#             --h5_file ${h5_file} \
#             --batch_size 10 \
#             --num_classes 50
# CUDA_VISIBLE_DEVICES=${device} python evaluation/fid/fid_example.py \
#         --image_folder results \
#         --h5_file ${h5_file} 

# SHAPE ALIGNED
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
# CUDA_VISIBLE_DEVICES=${device} python evaluation/neudist/neudist.py \
#                         --dataset birds \
#                         --testing_path results/${h5_file} \
#                         --model_name neudist_birds \
#                         --load_from_epoch 595

# BACKGROUND ALIGNED
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
# CUDA_VISIBLE_DEVICES=${device} python evaluation/neudist/neudist.py \
#                         --dataset birds \
#                         --testing_path results/${h5_file} \
#                         --model_name neudist_birds \
#                         --load_from_epoch 595

# NOTHING ALIGNED
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
# CUDA_VISIBLE_DEVICES=${device} python evaluation/neudist/neudist.py \
#                         --dataset birds \
#                         --testing_path results/${h5_file} \
#                         --model_name neudist_birds \
#                         --load_from_epoch 595


align='all'
name='SEG10_BG10L1_birds'
h5_file="$name""_G_epoch_""$epoch""_align_""$align""_background_noise"".h5"
python evaluation/ms_ssim/msssim_score.py \
        --image_folder results \
        --h5_file ${h5_file} \
        --evaluate_overall_score \
        --sample_per_cls 200

align='all' #tanto faz
name='BG10L1_birds'
h5_file="$name""_G_epoch_""$epoch""_align_""$align""_background_noise"".h5"
python evaluation/ms_ssim/msssim_score.py \
        --image_folder results \
        --h5_file ${h5_file} \
        --evaluate_overall_score \
        --sample_per_cls 200

align='all'
name='SEG10_birds'
h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
python evaluation/ms_ssim/msssim_score.py \
        --image_folder results \
        --h5_file ${h5_file} \
        --evaluate_overall_score \
        --sample_per_cls 200

align='all'
name='SEG10_BG10L1_char_birds'
h5_file="$name""_G_epoch_""$epoch""_align_""$align""_background_noise"".h5"
python evaluation/ms_ssim/msssim_score.py \
        --image_folder results \
        --h5_file ${h5_file} \
        --evaluate_overall_score \
        --sample_per_cls 200

align='all'
name='SEG10_BG10L1_0KL_birds'
h5_file="$name""_G_epoch_""$epoch""_align_""$align""_background_noise"".h5"
python evaluation/ms_ssim/msssim_score.py \
        --image_folder results \
        --h5_file ${h5_file} \
        --evaluate_overall_score \
        --sample_per_cls 200

align='none'
name='BASELINE_birds'
h5_file="$name""_G_epoch_""$epoch"".h5"
python evaluation/ms_ssim/msssim_score.py \
        --image_folder results \
        --h5_file ${h5_file} \
        --evaluate_overall_score \
        --sample_per_cls 200