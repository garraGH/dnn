/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/transform/transform.h
* author      : Garra
* time        : 2019-10-05 11:12:35
* description : 
*
============================================*/


#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"
#include "../rendererobject.h"

class Transform : public Asset, public std::enable_shared_from_this<Transform>
{
public:
    Transform(const std::string& name="unnamed") : Asset(name) {}
    static std::shared_ptr<Transform> Create(const std::string& name) { return std::make_shared<Transform>(name); }
    std::shared_ptr<Transform> Set(const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale) { m_translation = translation; m_rotation = rotation; m_scale = scale; m_dirty = true; return shared_from_this(); }

    std::shared_ptr<Transform> SetTranslation(const glm::vec3& translation) { m_translation = translation; m_dirty = true; return shared_from_this(); }
    std::shared_ptr<Transform> SetRotation(const glm::vec3& rotation) { m_rotation = rotation; m_dirty = true; return shared_from_this(); }
    std::shared_ptr<Transform> SetScale(const glm::vec3& scale) { m_scale = scale; m_dirty = true; return shared_from_this(); }
    const glm::vec3& GetTranslation() { return m_translation; }
    const glm::vec3& GetRotation() { return m_rotation; }
    const glm::vec3& GetScale() { return m_scale; }

    void Revert() { m_translation = glm::vec3(0.0f); m_rotation = glm::vec3(0.0f); m_scale = glm::vec3(1.0f); m_dirty = true; }
    void Translate(const glm::vec3& displacement) { m_translation += displacement; m_dirty = true; }
    void Rotate(const glm::vec3& degree) { m_rotation += degree; m_dirty = true; }
    void Scale(const glm::vec3& size) { m_scale += size; m_dirty = true; }

    const glm::mat4& GetTransformMatrx();

    void Debug();

protected:
    void _UpdateMatrix();

private:
    bool m_dirty = true;
    glm::vec3 m_translation = glm::vec3(0.0f);
    glm::vec3 m_rotation = glm::vec3(0.0f);
    glm::vec3 m_scale = glm::vec3(1.0f);
    glm::mat4 m_matTransform = glm::mat4(1.0f);
};
