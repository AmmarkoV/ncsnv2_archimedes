#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#CUDA_VISIBLE_DEVICES="0"

if (( $#<2 ))
then 
 echo "Please provide arguments first argument is configuration ,  second is experiment name/label "
 exit 1
else
 CONFIGURATION=$1
 EXPERIMENT=$2
fi

source ncsnv2Env/bin/activate
CUDA_VISIBLE_DEVICES="1" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python main.py --config $CONFIGURATION --doc $EXPERIMENT

cd "exp/logs"
scp -P 2222 -r $EXPERIMENT/ ammar@ammar.gr:/home/ammar/public_html/ncsnv2_archimedes/exp/logs/
echo "scp -P 2222 -r exp/logs/$EXPERIMENT/ ammar@ammar.gr:/home/ammar/public_html/ncsnv2_archimedes/exp/logs/"

exit 0
