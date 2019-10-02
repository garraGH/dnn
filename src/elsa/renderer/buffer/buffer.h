/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : buffer.h
* author      : Garra
* time        : 2019-10-01 22:13:27
* description : 
*
============================================*/


#pragma once
#include <vector>
#include <string>
#include "../rendererobject.h"
#include "../../core.h"

class Buffer : public RenderObject
{
public:
    class Element
    {
    public:
        enum class DataType { UnKnown = 0, Float, Float2, Float3, Float4, Mat3, Mat4, Int, Int2, Int3, Int4, UChar, UShort, UInt, Bool };

    public:
        Element(DataType type, bool normalized=false, const std::string& name="");
        DataType Type() const { return m_type; }
        bool Normalized() const { return m_normalized; }
        unsigned int Offset() const { return m_offset; }
        unsigned int Size() const ;
        unsigned int Components() const;

    private:
        DataType m_type = DataType::UnKnown;
        bool m_normalized = false;
        std::string m_name;
        unsigned int m_offset = 0;
        friend class Buffer;
    };

    class Layout
    {
    public:
        Layout() {}
        Layout(const std::initializer_list<Element>& elements);
        std::vector<Element>::const_iterator begin() const { return  m_elements.begin(); }
        std::vector<Element>::const_iterator end() const { return m_elements.end(); }
        unsigned int Stride() const { return m_stride; }
        void Push(Element& element);

    protected:
        void _calculateOffsetAndStride(Element& e);

    private:
        std::vector<Element> m_elements;
        unsigned int m_stride = 0;
    };
    
public:
    Buffer(unsigned int size);
    virtual ~Buffer();

    void SetLayout(const Layout& layout);
    unsigned int GetCount() const;
    

    static Buffer* CreateVertex(unsigned int size, float* data);
    static Buffer* CreateIndex(unsigned int size, void* data);

protected:
    virtual void _ApplyLayout() const;

protected:
    Layout m_layout;
    unsigned int m_size = 0;
};
