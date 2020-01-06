/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/core.c
* author      : Garra
* time        : 2019-12-13 16:42:31
* description : 
*
============================================*/


#include "core.h"
#include "glad/gl.h"

void GLClearError()
{
    while(glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* function, const char* file, int line)
{
    while(GLenum error = glGetError())
    {
        CORE_ERROR("[OpenGL Error]");
        switch(error)
        {
            case GL_INVALID_ENUM: CORE_ERROR("GL_INVALID_ENUM"); break;
            case GL_INVALID_VALUE: CORE_ERROR("GL_INVALID_VALUE"); break;
            case GL_INVALID_OPERATION: CORE_ERROR("GL_INVALID_OPERATION"); break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: CORE_ERROR("GL_INVALID_FRAMEBUFFER_OPERATION"); break;
            case GL_STACK_OVERFLOW: CORE_ERROR("GL_STACK_OVERFLOW"); break;
            case GL_STACK_UNDERFLOW: CORE_ERROR("GL_STACK_UNDERFLOW"); break;
            case GL_OUT_OF_MEMORY: CORE_ERROR("GL_OUT_OF_MEMORY"); break;
            default: CORE_ERROR(error);
        }
        CORE_ERROR("         File: {}", file);
        CORE_ERROR("     Function: {}", function);
        CORE_ERROR("         Line: {}", line);
        return false;
    }
    return true;
}
