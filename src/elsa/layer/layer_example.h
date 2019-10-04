/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layer_example.h
* author      : Garra
* time        : 2019-09-26 22:15:00
* description : 
*
============================================*/


#pragma once
#include "layer.h"
#include "logger.h"
#include "imgui.h"
#include "../renderer/shader/shader.h"
#include "../renderer/buffer/buffer.h"
#include "../renderer/camera/camera.h"



class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    void OnEvent(Event& e) override;
    void OnUpdate(float deltaTime) override;
    void OnImGuiRender() override;
 
protected:
    void _UpdateCamera(float deltaTime);
    void _UpdateScene();

private:
    std::shared_ptr<BufferArray> m_bufferArrayTri = nullptr;
    std::shared_ptr<BufferArray> m_bufferArrayQuad = nullptr;
    std::shared_ptr<Camera> m_camera = nullptr;

private:
    float m_speedTranslate = 0.5f;
    float m_speedRotate = 30.0f;
};
