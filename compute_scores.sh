# env = hdgan2
name='SEG10_BG50L2_birds'
# name='SEG10_BG10L1_IDTLOSS'
epoch='600'
device=2
export CUDA_VISIBLE_DEVICES=${device}

#### IS and FID
align='all'
h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
python evaluation/inception_score/inception_score.py \
            --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50
python evaluation/fid/fid_example.py \
        --image_folder results \
        --h5_file ${h5_file}            

align='shape'
h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
python evaluation/inception_score/inception_score.py \
            --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50
python evaluation/fid/fid_example.py \
        --image_folder results \
        --h5_file ${h5_file}

align='background'
h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
python evaluation/inception_score/inception_score.py \
            --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50
python evaluation/fid/fid_example.py \
        --image_folder results \
        --h5_file ${h5_file}

align='none'
h5_file="$name""_G_epoch_""$epoch""_align_$align"".h5"
python evaluation/inception_score/inception_score.py \
            --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50            
python evaluation/fid/fid_example.py \
        --image_folder results \
        --h5_file ${h5_file}