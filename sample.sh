name="SEG10_BG50L2"
# name="SEG10_BG10L1_IDTLOSS"
epoch="600"
device=2

CUDA_VISIBLE_DEVICES=${device} python sampling/sampler.py --model ${name} \
                                                          --epoch ${epoch} \
                                                          --batch_size 160 \
                                                          --align all \
                                                          --fix_seed