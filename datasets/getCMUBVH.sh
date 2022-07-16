#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
mkdir exp
cd exp
mkdir datasets
cd datasets


if [ -f cmubvh ]
then
echo "CMU BVH datasets appear to have been downloaded.."
else
  wget http://ammar.gr/datasets/archimedes.zip
  unzip archimedes.zip
  mv dataset cmubvh
  cd "$DIR" 
fi


exit 0
