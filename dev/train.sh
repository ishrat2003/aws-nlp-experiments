#!/bin/sh

#image=$1

image="aws-contextual-summary"

mkdir -p code/output
mkdir -p log

rm -R code/output/*

docker run -v $(pwd)/code:/opt/ml \
    -v /home/esw/ishrat/data/:/opt/ml/input/data/training/ \
    -v /home/esw/ishrat/output/:/opt/ml/output/ \
    --rm ${image} \
    | tee $(pwd)/log/server.log


#docker logs -t ${image}
