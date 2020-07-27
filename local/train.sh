#!/bin/sh

#image=$1

image="aws-contextual-summary"

mkdir -p code/output
mkdir -p code/log

rm -R code/output/*
rm code/log/*

mkdir -p code/output/data

docker run -v $(pwd)/code:/opt/ml --rm ${image} | tee $(pwd)/log/server.log
