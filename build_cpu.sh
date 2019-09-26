#=============================================
# Copyright(C)2019 Garra. All rights reserved.
# 
# file        : build_cpu.sh
# author      : Garra
# time        : 2019-09-24 11:11:05
# description : 
#
#=============================================


#!/bin/bash

xmake -D -v -F xmake_cpu.lua &&
xmake i -o ./ sandbox &&
./bin/sandbox
