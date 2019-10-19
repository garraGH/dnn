/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : ./shadertoy.cpp
* author      : Garra
* time        : 2019-10-13 10:48:15
* description : 
*
============================================*/


#include "shadertoy.h"
#include "shapes.h"
#include "shaping_functions.h"
#include "flatcolor.h"
#include "colors.h"
#include "matrix.h"
#include "hud.h"
#include "pattern.h"

std::shared_ptr<ShaderToy> ShaderToy::Create(Type type)
{
#define CASE(type) case Type::type: return type::Create()
    switch(type)
    {
        CASE(ShapingFunctions);
        CASE(FlatColor);
        CASE(Shapes);
        CASE(Colors);
        CASE(Matrix);
        CASE(HUD);
        CASE(Pattern);
        default: return nullptr;
    }
#undef CASE
}
