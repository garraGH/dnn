/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : camera.h
* author      : Garra
* time        : 2019-10-03 20:18:41
* description : 
*
============================================*/


#pragma once
#include "memory"
#include "glm/glm.hpp"
#include "../../event/event_key.h"
#include "../../event/event_mouse.h"
#include "../../event/event_application.h"

class Camera
{
public:
    enum class Type
    {
        Orthographic, 
        Perspective 
    };

    enum class Usage
    {
        TwoDimension, 
        ThreeDimension, 
    };

    class Sight
    {
    public:
        void Set(float widthOrVfov, float asp, float near, float far) { m_type == Type::Orthographic? m_width = widthOrVfov : m_vfov = widthOrVfov; m_asp = asp; m_near = near; m_far = far; m_dirty = true; }
        void SetType(Type t)            { m_type = t;       m_dirty = true; }
        void SetWidth(float width)      { m_width = width;  m_dirty = true; }
        void SetVfov(float vfov)        { m_vfov = vfov;    m_dirty = true; }
        void SetAspectRatio(float asp)  { m_asp = asp;      m_dirty = true; }
        void SetNear(float n)           { m_near = n;       m_dirty = true; }
        void SetFar(float f)            { m_far = f;        m_dirty = true; }
        void SetScale(float s)          { m_scale = s;      m_dirty = true; }
        void Scale(float ds)            { m_scale += ds;    m_dirty = true; }

        Type GetType() const    { return m_type;  }
        float GetWidth() const  { return m_width; }
        float GetVfov() const   { return m_vfov;  }
        float GetAsp() const    { return m_asp;   }
        float GetNear() const   { return m_near;  }
        float GetFar() const    { return m_far;   }
        float GetScale() const  { return m_scale; }
        bool IsDirty() const    { return m_dirty; }

        void OnImGuiRender();

        const glm::mat4& View2Clip() { _Update(); return m_matView2Clip; }
        const glm::mat4& Clip2View() { _Update(); return m_matClip2View; }
        float DepthClip2NDC(float depthInClip) const;
        float DepthNDC2Clip(float depthInNDC) const;

    protected:
        void _Update();
        void _calcOrthoMatrix();
        void _calcPerspectiveMatrix();

    private:
        Type m_type;
        float m_width = 2;      //ortho
        float m_vfov = 45;      //VerticalFieldOfView, perspective
        float m_asp = 1;        //aspect ratio = width/height
        float m_near = 1;
        float m_far = 101;
        bool m_dirty = true;
        float m_scale = 1.0;

        glm::mat4 m_matView2Clip = glm::mat4(1);
        glm::mat4 m_matClip2View = glm::mat4(1);
    };
    
public:
    Camera(const std::string& name, Type type=Type::Perspective, Usage usage=Usage::ThreeDimension) : m_name(name), m_usage(usage) { m_sight.SetType(type); }
    const std::string& GetName()    { return m_name;                }
    void SetUsage(Usage u)          { m_usage = u;                  }
    void SetType(Type t)            { m_sight.SetType(t);           }
    void SetWidth(float w)          { m_sight.SetWidth(w);          }
    void SetVfov(float vfov)        { m_sight.SetVfov(vfov);        }
    void SetAspectRatio(float asp)  { m_sight.SetAspectRatio(asp);  }
    void SetNear(float n)           { m_sight.SetNear(n);           }
    void SetFar(float f)            { m_sight.SetFar(f);            }
    void SetSight(float widthOrVfov, float asp, float near, float far) { m_sight.Set(widthOrVfov, asp, near, far); }

    inline const glm::vec3 GetDirection() const     { return glm::normalize(m_target-m_position); }
    inline const glm::vec3& GetPosition() const     { return m_position;                   }
    inline const glm::vec3& GetTarget() const       { return m_target;                     }
    inline const glm::vec3& GetOrientation() const  { return m_orientation;                }
    inline float GetScale() const                   { return m_sight.GetScale();           }
    inline void SetPosition(const glm::vec3& pos)   { m_position = pos; m_dirty = 0x01010101;    }
    inline void SetTarget(const glm::vec3& target)  { m_target = target; m_dirty = 0x01010101;   }
    inline void SetOrientation(const glm::vec3& ori){ m_orientation = ori; m_dirty = 0x01010101; }

    inline void SetTranslationSpeed(float speed){ m_speedTrans = speed; }
    inline void SetRotationSpeed(float speed)   { m_speedRotat = speed; }
    inline void SetScaleSpeed(float speed)      { m_speedScale = speed; }

    inline void Translate(const glm::vec3& dis) { m_position += dis; m_target += dis; m_dirty = 0x01010101;      }
    inline void Rotate(const glm::vec3& angle)  { m_orientation += angle; m_dirty = 0x01010101; }
    inline void Scale(float s)                  { m_sight.Scale(s);                       }
    inline void Revert()                        { m_position = glm::vec3(0, 0, 10); m_target = glm::vec3(0); m_orientation = glm::vec3(0.0f); m_sight.SetScale(1.0); m_dirty = 0x01010101; }

    inline const glm::mat4& World2View() { _UpdateWorld2View(); return m_matWorld2View; }
    inline const glm::mat4& View2World() { _UpdateView2World(); return m_matView2World; }
    inline const glm::mat4& View2Clip()  { return m_sight.View2Clip();                  }
    inline const glm::mat4& Clip2View()  { return m_sight.Clip2View();                  }
    inline const glm::mat4& World2Clip() { _UpdateWorld2Clip(); return m_matWorld2Clip; }
    inline const glm::mat4& Clip2World() { _UpdateClip2World(); return m_matClip2World; }

    glm::vec3 Screen2World(const glm::vec2& pntOnScreen);
    glm::vec2 World2Screen(const glm::vec3& posInWorld);
    glm::vec3 Screen2View(const glm::vec2& pntOnScreen);
    glm::vec2 View2Screen(const glm::vec3& posInView);
    glm::vec3 View2World(const glm::vec3& posInView);
    glm::vec3 World2View(const glm::vec3& posInWorld);

    std::array<glm::vec3, 4> GetNearCornersInWorldSpace();
    std::array<glm::vec3, 4> GetFarCornersInWorldSpace();

    glm::vec3 GetPosFromClipSpaceToWorldSpace(const glm::vec3& pos);
    inline bool NeedUpdate() const              { return m_sight.IsDirty() || m_dirty != 0; }
    inline void EnableRotation(bool enabled)    { m_rotationEnabled = enabled;         }

    void OnUpdate(float deltaTime);
    void OnEvent(Event& e);
    void OnImGuiRender(bool independent=true);

    static std::shared_ptr<Camera> Create(const std::string& name, Type type=Type::Perspective, Usage usage=Usage::ThreeDimension);

protected:
    void _UpdateWorld2View();
    void _UpdateView2World();
    void _UpdateWorld2Clip();
    void _UpdateClip2World();
    std::array<glm::vec3, 4> _GetCornersInWorldSpace(float d);
    void _ShowPosTooptips();

protected:
    bool _OnMouseScrolled(MouseScrolledEvent& e);
    bool _OnMouseButtonPressed(MouseButtonPressedEvent& e);
    bool _OnMouseButtonReleased(MouseButtonReleasedEvent& e);
    bool _OnMouseMoved(MouseMovedEvent& e);
    bool _OnWindowResize(WindowResizeEvent& e);
    void _DragScene();
    void _RotateAroundPosOfButtonPressed(MouseMovedEvent& e);
    glm::vec3 _RotateAroundTargetOnHorizontalPlane(const glm::vec3& pos, const glm::vec3& target, float theta);
    glm::vec3 _RotateAroundTargetOnVerticalPlane(const glm::vec3& pos, const glm::vec3& target, float theta);
    glm::vec3 _GetWorldPosOfCurrentCursorOnDragPlane();
    glm::vec3 _GetWorldPosOfCurrentCursorOnHorizontalPlane();
    glm::vec3 _GetWorldPosOfCurrentCursorOnVerticalPlane();

private:
    std::string m_name;
    union
    {
        int m_dirty = 0x01010101;
        struct
        {
            bool m_dirtyW2V;
            bool m_dirtyV2W;
            bool m_dirtyW2C;
            bool m_dirtyC2W;
        };
    };

    bool m_rotationEnabled = true;
    bool m_targetFixed = true;
    Type m_type = Type::Perspective;
    Usage m_usage = Usage::ThreeDimension;
    Sight m_sight;

    glm::mat4 m_matWorld2View = glm::mat4(1.0f);
    glm::mat4 m_matView2World = glm::mat4(1.0f);
    glm::mat4 m_matWorld2Clip = glm::mat4(1.0f);
    glm::mat4 m_matClip2World = glm::mat4(1.0f);

    glm::vec3 m_position = glm::vec3(0, 10, 10);
    glm::vec3 m_target = glm::vec3(0);
    glm::vec3 m_up = glm::vec3(0, 1, 0);
    glm::vec3 m_orientation = glm::vec3(0);

    float m_speedTrans = 1;
    float m_speedRotat = 20.0f;
    float m_speedScale = 0.2f;

    bool m_bLeftButtonPressed = false;
    bool m_bRightButtonPressed = false;
    std::pair<float, float> m_cursorScreenPntWhenButtonPressed;
    glm::vec3 m_worldPosOfCursorWhenButtonPressed;
    glm::vec3 m_cameraWorldPositionWhenButtonPressed;
    glm::vec3 m_cameraWorldTargetWhenButtonPressed;

    std::array<unsigned int, 2> m_windowSize = {1000, 1000};
};
