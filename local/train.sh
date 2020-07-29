#!/bin/sh

#image=$1

image="aws-contextual-summary"

mkdir -p code/output
mkdir -p log

rm -R code/output/*

docker run -v $(pwd)/code:/opt/ml \
    -v /Users/ishratsami/Workspace/research_projects/data/:/opt/ml/input/data/training/ \
    -v /Users/ishratsami/Workspace/research_projects/output/:/opt/ml/output/ \
    --rm ${image} \
    | tee $(pwd)/log/server.log


#docker logs -t ${image}
