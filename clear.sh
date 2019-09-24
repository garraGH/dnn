#=============================================
# Copyright(C)2019 Garra. All rights reserved.
# 
# file        : clear.sh
# author      : Garra
# time        : 2019-09-24 11:06:49
# description : 
#
#=============================================


#!/bin/bash


find ./ -type d -name "build" -exec rm -r {} +
find ./ -type d -name "\.xmake" -exec rm -r {} +
rm -rf ./bin

