name="model5"
# epoch="550"
epoch="390"
device=0

# text, shape, background

# for i in $(seq 1 10)
# do
CUDA_VISIBLE_DEVICES=${device} python sampling/1Dinterpolation.py --model ${name} \
                                                                  --epoch ${epoch} \
                                                                  --nb_interp 5 \
                                                                  --interpolate text \
                                                                  --fx1_id 121 \
                                                                  --fx2_id 121 \
                                                                  --mv_ida 121 \
                                                                  --mv_idb 96
# done
# CUDA_VISIBLE_DEVICES=${device} python sampling/2Dinterpolation.py --model ${name} \
#                                                                   --epoch ${epoch} \
#                                                                   --nb_interp 5 \
#                                                                   --fix shape \
#                                                                   --fx_id 66 \
#                                                                   --mv2_ida 26 \
#                                                                   --mv1_idb 2 \
#                                                                   # --mv1_ida 52 \
#                                                                   # --mv2_idb 52 

# CUDA_VISIBLE_DEVICES=${device} python sampling/3Dinterpolation.py --model ${name} \
#                                                                   --epoch ${epoch} \
#                                                                   --nb_interp 5 \
#                                                                   # --mv1_ida 70 \
#                                                                   # --mv1_idb 68 \
#                                                                   # --mv2_ida 66 \
#                                                                   # --mv2_idb 71 \
#                                                                   # --mv3_idb 71 \
#                                                                   # --mv3_idb 71                                                               