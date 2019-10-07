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
#include "glm/gtc/type_ptr.hpp"



class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    void OnEvent(Event& e) override;
    void OnUpdate(float deltaTime) override;
    void OnImGuiRender() override;
 
protected:
    void _UpdateScene(float deltaTime);
    void _UpdateCamera(float deltaTime);
    void _UpdateTri(float deltaTime);
    void _UpdateQuads(float deltaTime);
    void _PrepareAssets();

private:
    std::shared_ptr<Renderer::Element> m_reTri = nullptr;
    std::shared_ptr<Renderer::Element> m_reQuad = nullptr;
    std::shared_ptr<Camera> m_camera = nullptr;
    float m_speedTranslate = 0.5f;
    float m_speedRotate = 30.0f;
    std::shared_ptr<Transform> m_transformTri = std::make_shared<Transform>();
    std::shared_ptr<Transform> m_transformQuad = std::make_shared<Transform>(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.1f));
};
