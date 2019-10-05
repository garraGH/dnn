/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : transform.cpp
* author      : Garra
* time        : 2019-10-05 11:12:34
* description : 
*
============================================*/


#include "transform.h"
#include "logger.h"
#include "glm/gtx/string_cast.hpp"

Transform::Transform(const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale)
    : m_translation(translation)
    , m_rotation(rotation)
    , m_scale(scale)
{

}

const glm::mat4& Transform::GetTransformMatrx()
{
    if(m_dirty)
    {
        _UpdateMatrix();
    }

    return m_matTranform;
}

void Transform::_UpdateMatrix()
{
//     m_matTranform = glm::scale(glm::rotate(glm::rotate(glm::rotate(glm::translate(glm::mat4(1.0f), m_translation), glm::radians(m_rotation.x), glm::vec3(1, 0, 0)), glm::radians(m_rotation.y), glm::vec3(0, 1, 0)), glm::radians(m_rotation.z), glm::vec3(0, 0, -1)), m_scale); 
//     m_matTranform = glm::translate(glm::rotate(glm::rotate(glm::rotate(glm::scale(glm::mat4(1.0f), m_scale), glm::radians(m_rotation.x), glm::vec3(1, 0, 0)), glm::radians(m_rotation.y), glm::vec3(0, 1, 0)), glm::radians(m_rotation.z), glm::vec3(0, 0, -1)), m_translation); 
//     m_matTranform = glm::rotate(glm::rotate(glm::rotate(glm::translate(glm::scale(glm::mat4(1.0f), m_scale), m_translation), glm::radians(m_rotation.x), glm::vec3(1, 0, 0)), glm::radians(m_rotation.y), glm::vec3(0, 1, 0)), glm::radians(m_rotation.z), glm::vec3(0, 0, -1)); 
//     m_matTranform = glm::rotate(glm::rotate(glm::rotate(glm::scale(glm::translate(glm::mat4(1.0f), m_translation), m_scale), glm::radians(m_rotation.x), glm::vec3(1, 0, 0)), glm::radians(m_rotation.y), glm::vec3(0, 1, 0)), glm::radians(m_rotation.z), glm::vec3(0, 0, -1)); 
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), m_translation);
    glm::mat4 rotationX = glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.x), glm::vec3(1, 0, 0));
    glm::mat4 rotationY = glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.y), glm::vec3(0, 1, 0));
    glm::mat4 rotationZ = glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.z), glm::vec3(0, 0, 1));
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), m_scale);
//     m_matTranform = rotationX*rotationY*rotationZ*translation*scale;
//     m_matTranform = scale*translation*rotationX*rotationY*rotationZ;
    m_matTranform = translation*rotationX*rotationY*rotationZ*scale;
//     m_matTranform = scale*rotationX*rotationY*rotationZ*translation;

    m_dirty = false;
}

void Transform::Debug()
{
    INFO("translation: {}", glm::to_string(m_translation));
    INFO("rotation: {}", glm::to_string(m_rotation));
    INFO("scale: {}", glm::to_string(m_scale));
}
