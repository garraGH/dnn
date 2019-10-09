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
#include "../renderer/renderer.h"



class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    void OnEvent(Event& e) override;
    void OnUpdate(float deltaTime) override;
    void OnImGuiRender() override;
 
protected:
    bool _OnKeyPressed(KeyPressedEvent& e);
    void _UpdateScene(float deltaTime);
    void _UpdateCamera(float deltaTime);
    void _UpdateTri(float deltaTime);
    void _UpdateQuads(float deltaTime);
    void _TransformQuads(float deltaTime);
    void _PrepareResources();

private:
    std::unique_ptr<CameraContoller> m_cameraController = std::make_unique<CameraContoller>(Camera::Type::Orthographic);
    float m_speedTranslate = 0.5f;
    float m_speedRotate = 30.0f;
};
