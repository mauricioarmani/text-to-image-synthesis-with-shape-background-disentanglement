# name="SEG10_BG50L2"
name="SEG10_BG10L1"
# name="SEG10_BG10L1_0KL"
epoch="500"
device=2

CUDA_VISIBLE_DEVICES=${device} python evaluation/ms_ssim/msssim_score_txt_consistency.py --model ${name} \
                                                          --epoch ${epoch} \
                                                          --batch_size 100 \
                                                          --align all \
                                                          # --fix_seed
