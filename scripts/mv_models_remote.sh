#!/bin/bash

SRC=CAF78_day1
DEST=CAF78_day2

alias s3='aws --endpoint https://s3-central.nrp-nautilus.io s3'
for model in $(s3 ls s3://hengenlab/${SRC}/Runs/ | awk '{print $2}');do 
    s3 cp s3://hengenlab/${SRC}/Runs/${model}Model s3://hengenlab/${DEST}/Runs/${model}Model/ --recursive; 
done