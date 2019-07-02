# MELHOR EPOCA = 500 (melhor FID e segundo melhor IS por pouco)
name='SEG10_BG50L2_birds'
# name='SEG10_BG10L1_IDTLOSS_birds'

epoch=500

# SHAPE ALIGNED
CUDA_VISIBLE_DEVICES="2" python gan/test_worker.py \
    --dataset birds \
    --model_name ${name} \
    --load_from_epoch ${epoch} \
    --test_sample_num 10 \
    --save_visual_results \
    --batch_size 64 \
    --align shape

# BACKGROUND ALIGNED
CUDA_VISIBLE_DEVICES="2" python gan/test_worker.py \
    --dataset birds \
    --model_name ${name} \
    --load_from_epoch ${epoch} \
    --test_sample_num 10 \
    --save_visual_results \
    --batch_size 64 \
    --align background

# ALL ALIGNED
CUDA_VISIBLE_DEVICES="2" python gan/test_worker.py \
    --dataset birds \
    --model_name ${name} \
    --load_from_epoch ${epoch} \
    --test_sample_num 10 \
    --save_visual_results \
    --batch_size 64 \
    --align all \

# ALL ALIGNED AND RANDOM SHAPE NOISE
CUDA_VISIBLE_DEVICES="2" python gan/test_worker.py \
    --dataset birds \
    --model_name ${name} \
    --load_from_epoch ${epoch} \
    --test_sample_num 10 \
    --save_visual_results \
    --batch_size 64 \
    --align all \
    --shape_noise

# ALL ALIGNED AND RANDOM BACKGROUND NOISE
CUDA_VISIBLE_DEVICES="2" python gan/test_worker.py \
    --dataset birds \
    --model_name ${name} \
    --load_from_epoch ${epoch} \
    --test_sample_num 10 \
    --save_visual_results \
    --batch_size 64 \
    --align all \
    --background_noise

# ALL ALIGNED AND RANDOM BACKGROUND AND SHAPE NOISES
CUDA_VISIBLE_DEVICES="2" python gan/test_worker.py \
    --dataset birds \
    --model_name ${name} \
    --load_from_epoch ${epoch} \
    --test_sample_num 10 \
    --save_visual_results \
    --batch_size 64 \
    --align all \
    --shape_noise \
    --background_noise  

# NOTHING ALIGNED
CUDA_VISIBLE_DEVICES="2" python gan/test_worker.py \
    --dataset birds \
    --model_name ${name} \
    --load_from_epoch ${epoch} \
    --test_sample_num 10 \
    --save_visual_results \
    --batch_size 64 \
    --align none 