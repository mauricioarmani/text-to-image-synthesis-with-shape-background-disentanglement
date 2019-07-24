# name="SEG10_BG50L2"
# name="SEG10_BG10L1"
# name="SEG10_BG10L1_0KL"
epoch="500"
device=0

CUDA_VISIBLE_DEVICES=${device} python sampling/sampler.py --model ${name} \
                                                          --epoch ${epoch} \
                                                          --batch_size 160 \
                                                          --align none \
                                                          # --fix_seed

# CUDA_VISIBLE_DEVICES=${device} python sampling/shape_consistency.py --model ${name} \
#                                                           --epoch ${epoch} \
#                                                           --batch_size 80 \
#                                                           --align none \

# CUDA_VISIBLE_DEVICES=${device} python sampling/txt_consistency.py --model ${name} \
#                                                           --epoch ${epoch} \
#                                                           --batch_size 80 \
#                                                           --align none

# CUDA_VISIBLE_DEVICES=${device} python sampling/bkg_consistency.py --model ${name} \
#                                                           --epoch ${epoch} \
#                                                           --batch_size 80 \
#                                                           --align none