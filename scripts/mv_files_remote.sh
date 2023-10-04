#!/bin/bash
# First do 'conda activate tovi_dl'

ANIMAL=CAF61_day1
ENDPOINT=https://s3-central.nrp-nautilus.io
DIR=/media/bs007r/CAF00061/CAF00061_2020-12-07_17-41-34/ #always put '/' in the end
DEST=s3://hengenlab/${ANIMAL}/Neural_Data/
FIRST=288

files_to_move=$(ls ${DIR} | grep Head | head -$FIRST)
#files_to_move=$(ls ${DIR} | grep Head | tail -$FIRST)
files_moved=$(aws --endpoint $ENDPOINT s3 ls ${DEST} | awk '{print $4}')

for file in $files_to_move; do
    if [[ "$file" == *bin* ]] && [[ ! $files_moved =~ "$file" ]]; then
        s4cmd --endpoint $ENDPOINT put -c 20 "$DIR$file" "$DEST$file"
    else
        echo $file exists or not a binary files
    fi
done
