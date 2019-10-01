/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : buffer_opengl.cpp
* author      : Garra
* time        : 2019-10-01 22:13:42
* description : 
*
============================================*/


#include "buffer_opengl.h"
#include "glad/gl.h"

OpenGLBuffer::OpenGLBuffer()
{
    glGenBuffers(1, &m_id);
}

OpenGLBuffer::~OpenGLBuffer()
{
    glDeleteBuffers(1, &m_id);
}

OpenGLVertexBuffer::OpenGLVertexBuffer(unsigned int size, float* data)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    m_count = size/sizeof(float);
}

void OpenGLVertexBuffer::Bind(unsigned int slot) const
{
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
}

void OpenGLVertexBuffer::Unbind() const
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

OpenGLIndexBuffer::OpenGLIndexBuffer(unsigned int size, unsigned int* data)
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    m_count = size/sizeof(unsigned int);
}


void OpenGLIndexBuffer::Bind(unsigned int slot) const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id);
}

void OpenGLIndexBuffer::Unbind() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

