#!/bin/sh
#wget https://pjreddie.com/media/files/train2014.zip
#wget tar xzf labels.tgz
#unzip train2014.zip -d ../
#tar xzf labels.tgz -C ../
mkdir ../data
mv ../labels/train2014 ../data/labels
mv ../train2014 ../data/images
rm -r ../labels
