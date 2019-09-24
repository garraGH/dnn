#=============================================
# Copyright(C)2019 Garra. All rights reserved.
# 
# file        : build.sh
# author      : Garra
# time        : 2019-09-24 11:11:05
# description : 
#
#=============================================


#!/bin/bash


xmake p -D -v -o pkg/build utils &&
xmake p -D -v -o pkg/build elsa  &&
xmake -D -v -F xmake.lua sandbox &&
xmake i -o ./ sandbox            &&
./bin/sandbox
