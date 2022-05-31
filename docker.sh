docker run -it --runtime=nvidia  --rm \
-v $(realpath ~/coding/sparse-anova/notebooks):/tf/notebooks \
-p 8888:8888 \
-e PASSWORD=sparse \
tensorflow/tensorflow:latest-gpu-jupyter