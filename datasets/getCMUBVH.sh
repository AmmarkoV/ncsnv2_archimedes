#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
mkdir datasets
cd datasets


if [ -f CMUBVH ]
then
echo "CMU BVH datasets appear to have been downloaded.."
else
  wget http://ammar.gr/datasets/archimedes.zip
  unzip archimedes.zip
  mv dataset CMUBVH
  cd "$DIR" 
fi


exit 0
