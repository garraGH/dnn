/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : renderer.h
* author      : Garra
* time        : 2019-10-01 23:14:38
* description : 
*
============================================*/


#pragma once

enum class RendererAPI
{
    UNKOWN = 0, 
    OpenGL = 1, 
    Vulcan = 2,
    DirectX9 = 3, 
    DirectX11 = 4, 
    Directx12 = 5, 
    Metal = 6
};

class Renderer
{
public:
    inline static RendererAPI GetAPI() { return s_api; }
    inline static void SetAPI(RendererAPI api) { s_api = api; }
        
private:
    static RendererAPI s_api;
};
