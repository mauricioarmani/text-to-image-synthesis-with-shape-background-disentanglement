# name="SEG10_BG50L2"
name="SEG10_BG10L1"
epoch="500"
device=1

# text, shape, background
# for i in $(seq 100)
# do
# CUDA_VISIBLE_DEVICES=${device} python sampling/1Dinterpolation.py --model ${name} \
#                                                                   --epoch ${epoch} \
#                                                                   --nb_interp 5 \
#                                                                   --interpolate background \
#                                                                   # --fx1_id 121 \
#                                                                   # --fx2_id 121 \
#                                                                   # --mv_ida 121 \
#                                                                   # --mv_idb 96
# done

# for i in $(seq 1 100)
# do
# CUDA_VISIBLE_DEVICES=${device} python sampling/2Dinterpolation.py --model ${name} \
#                                                                   --epoch ${epoch} \
#                                                                   --nb_interp 5 \
#                                                                   --fix background \
#                                                                   # --fx_id 138 \
#                                                                   # --mv2_ida 79 \
#                                                                   # --mv1_idb 79 \
#                                                                   # --mv1_ida 92 \
#                                                                   # --mv2_idb 92 
# done

for i in $(seq 1 100)
do
CUDA_VISIBLE_DEVICES=${device} python sampling/3Dinterpolation.py --model ${name} \
                                                                  --epoch ${epoch} \
                                                                  --nb_interp 5 \
                                                                  # --mv1_ida 70 \
                                                                  # --mv1_idb 68 \
                                                                  # --mv2_ida 66 \
                                                                  # --mv2_idb 71 \
                                                                  # --mv3_idb 71 \
                                                                  # --mv3_idb 71                                                               
done