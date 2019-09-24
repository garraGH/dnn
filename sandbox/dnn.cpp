/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : dnn.cpp
* author      : Garra
* time        : 2019-09-24 10:48:21
* description : 
*
============================================*/


#include "elsa.h"

class DNN : public Application
{
public:
    DNN()
    {

    }

    ~DNN()
    {

    }
protected:
private:
};

Application* CreateApplication()
{
    return new DNN();
}
