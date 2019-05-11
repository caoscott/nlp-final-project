#!/bin/sh

wget http://images.cocodataset.org/zips/train2014.zip -q --show-progress
wget http://images.cocodataset.org/zips/val2014.zip -q --show-progress
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -q --show-progress
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -q --show-progress
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -q --show-progress
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -q --show-progress

unzip -q '*.zip'
