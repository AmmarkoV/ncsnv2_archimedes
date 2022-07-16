#!/bin/bash

#Reminder , to get a single file out of the repo use  "git checkout -- path/to/file.c"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd .. 

#Now sync rest of code
cd "$DIR"
git reset --hard HEAD
git pull origin master
  
exit 0
