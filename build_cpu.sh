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


xmake p -D -v -o pkg/build -F xmake_cpu.lua utils &&
xmake p -D -v -o pkg/build -F xmake_cpu.lua elsa  &&
xmake -D -v -F xmake_cpu.lua sandbox &&
xmake i -o ./ sandbox            &&
./bin/sandbox
