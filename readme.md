## Text-to-Image Synthesis with Shape and Background Disentanglement (Pytorch implementation)

<!-- # TODO: arxiv -->
<!-- > [Zizhao Zhang*, Yuanpu Xie*, Lin Yang, "Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network", CVPR (2018)](https://arxiv.org/abs/1802.09178) * indicates contribution -->

<!-- # TODO: figura
<p align="center">
  <img src ="Figures/arch.jpg" width="1200px" />
</p>
<p align="center" >
Visual results (Left: compared against StackGAN; Right: multi-resolution generator outputs)
  <img src ="Figures/samples.png" width="1200px" />
</p> -->

## Dependencies
- Python 3
- Pytorch
- Anaconda 3.6
- Tensorflow 1.4.1 (for evaluation only)

## Data
Download preprocessed data in /Data.
- Download [birds](https://www.dropbox.com/sh/v0vcgwue2nkwgrf/AACxoRYTAAacmPVfEvY-eDzia?dl=0) to Data/birds

## Training
- `sh train_birds.sh`


### Monitor your training with tensorboardX
- tensorboard `--logdir models/`

## Testing
- `sh test_birds.sh`

## Evaluation
Evaluation needs the sampled results obtained in Testing and saved in ./Results.

Inception score
- Download [inception models](https://www.dropbox.com/sh/lpzsvwabkw8d26g/AADFRKpTvbylhl0Q3PH78qzha?dl=0) to the evaluation/inception_score/inception_finetuned_models folder than run `sh compute_scores.sh`.


<!-- # TODO: figura -->
<!-- ## Pretrained Models -->
<!-- We provide pretrained models for birds, flowers, and coco. -->
<!-- - Download the [pretrained models](https://www.dropbox.com/sh/lpzsvwabkw8d26g/AADFRKpTvbylhl0Q3PH78qzha?dl=0). Save them to the Models/ folder. -->
<!-- - It contains HDGAN for birds and flowers, visual similarity model for birds and flowers -->


<!-- ## Acknowlegements -->
<!-- - Inception score [Tensorfow implementation](https://github.com/awslabs/deeplearning-benchmark/tree/master/tensorflow/inception) -->


<!-- ### Citation -->
<!-- If you find our work useful in your research, please cite: -->


<!-- ## License -->
<!-- MIT -->