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


xmake p -o ../../pkg/build -P pkg_src/elsa
xmake -P sandbox
xmake r -F sandbox/xmake.lua
