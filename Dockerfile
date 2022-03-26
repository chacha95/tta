FROM nvidia/cuda:10.1-cudnn7-devel
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -y python3-opencv ca-certificates python3-dev wget git unzip vim \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# install pip
ENV PATH="/root/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py \
	&& python3 get-pip.py --user \
	&& rm get-pip.py

# install torch
RUN pip install --no-cache-dir --user torch==1.8 torchvision==0.9 -f https://download.pytorch.org/whl/cu101/torch_stable.html

# install fvcore and detectron2
WORKDIR /home
ENV FORCE_CUDA="1"
ENV FVCORE_CACHE="/tmp"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --no-cache-dir --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install --no-cache-dir --user -e detectron2_repo

# install COCO2017 dataset
ENV COCO_DIR="/home/dataset/COCO2017"
ENV COCO_IMG="${COCO_DIR}/val2017.zip"
ENV COCO_ANNOTATION="${COCO_DIR}/annotations_trainval2017.zip"
RUN mkdir -p "/home/dataset/COCO2017"
RUN wget "http://images.cocodataset.org/zips/val2017.zip" -O "${COCO_IMG}" \
    && unzip "${COCO_IMG}" -d "${COCO_DIR}" && rm -rf "${COCO_IMG}"
RUN wget "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -O "${COCO_ANNOTATION}" \
    && unzip "${COCO_ANNOTATION}" -d "${COCO_DIR}" && rm -rf "${COCO_ANNOTATION}"

# COPY source code
COPY src /home/src
# pretrained model download
RUN wget "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl" \
     -O "/home/src/model/faster_rcnn_R_50_FPN_3x.pkl"
WORKDIR /home/src