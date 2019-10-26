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
#include "elsa.h"

class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    void OnEvent(Event& e) override;
    void OnUpdate(float deltaTime) override;
    void OnImGuiRender() override;
 
    static std::shared_ptr<ExampleLayer> Create();

protected:
    void _UpdateScene(float deltaTime);
    void _UpdateViewport(float deltaTime);
    void _UpdateTri(float deltaTime);
    void _UpdateQuads(float deltaTime);
    void _TransformQuads(float deltaTime);
    void _PrepareResources();
    void _PrepareViewports();

private:
    std::shared_ptr<Viewport> m_viewport[4];

    float m_speedTranslate = 0.5f;
    float m_speedRotate = 30.0f;
};
