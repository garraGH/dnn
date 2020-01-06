/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : sandbox/learnopengl/basicrenderelement.cpp
* author      : Garra
* time        : 2019-12-30 17:13:48
* description : 
*
============================================*/


#include "basicrenderelement.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "osdialog.h"
#include "stb_image.h"

//////////////////////////////////////////////
REQuad::REQuad(const std::string& name)
    : Renderer::Element(name)
{
    _Prepare();
}

void REQuad::_PrepareMesh()
{
    if(m_mesh != nullptr)
    {
        return;
    }

    float vertices[] = 
    {
        -1, -1, 0, 0, 0, 
        +1, -1, 0, 1, 0, 
        +1, +1, 0, 1, 1, 
        -1, +1, 0, 0, 1, 
    };

    unsigned char indices[] = 
    {
        0, 1, 2, 
        0, 2, 3, 
    };

    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout({ {Buffer::Element::DataType::Float3, "a_Position", false}, {Buffer::Element::DataType::Float2, "a_TexCoord", false} });
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout({ {Buffer::Element::DataType::UChar} });
    m_mesh = Renderer::Resources::Create<Elsa::Mesh>("mesh_Quad")->Set(ib, {vb});
}

void REQuad::_PrepareTexture()
{
    if(!Renderer::Resources::Exist<Texture2D>("t2d_BRDF"))
    {
        Renderer::Resources::Create<Texture2D>("t2d_BRDF");
    }
}

void REQuad::_PrepareMaterial()
{
    if(m_material == nullptr)
    {
        using MU = Material::Uniform;
        std::shared_ptr<MU> muMS2WS = Renderer::Resources::Create<MU>("mu_MS2WS")->Set(MU::Type::Mat4x4, 1, glm::value_ptr(Transform::Create("tr_Temp")->Translate(glm::vec3(0, 0, 2))->Get()));
        m_material = Renderer::Resources::Create<Material>("mtr_Quad");
        m_material->SetTexture("u_Texture", Renderer::Resources::Get<Texture2D>("t2d_BRDF"));
        m_material->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("ub_Transform"));
        m_material->SetUniform("u_MS2WS", muMS2WS);
    }
}

void REQuad::_PrepareShader()
{
    if(m_shader == nullptr)
    {
        m_shader = Renderer::Resources::Create<Shader>("shader_Quad")->LoadFromFile("/home/garra/study/dnn/assets/shader/Quad.glsl");
    }
}

//////////////////////////////////////////////
RESkybox::RESkybox(const std::string& name)
    : Renderer::Element(name)
{
    _Prepare();
}

void RESkybox::_PrepareMesh()
{
    if(m_mesh != nullptr)
    {
        return;
    }

    float vertices[] =  // pos stq
    {
        -1, -1, -1,
        +1, -1, -1,
        +1, +1, -1,
        -1, +1, -1,
                   
        -1, -1, +1,
        +1, -1, +1,
        +1, +1, +1,
        -1, +1, +1,
    };
    unsigned char indices[] = 
    {
        0, 1, 2, 
        0, 2, 3, 
        4, 0, 3, 
        4, 3, 7, 
        5, 4, 7, 
        5, 7, 6, 
        1, 5, 6, 
        1, 6, 2, 
        0, 4, 5, 
        0, 5, 1, 
        3, 2, 6, 
        3, 6, 7, 
    };

    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout({Buffer::Element::DataType::UChar});
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout({{Buffer::Element::DataType::Float3, "a_Position", false}});
    m_mesh = Renderer::Resources::Create<Elsa::Mesh>("mesh_Skybox")->Set(ib, {vb});
}


void RESkybox::_PrepareTexture()
{
    if(m_cubemap == nullptr)
    {
        m_cubemap = Renderer::Resources::Create<TextureCubemap>("tcm_Skybox")->Load("/home/garra/study/dnn/assets/texture/skybox/sunny/right.jpg", false);
    }
}

void RESkybox::_PrepareMaterial()
{
    if(m_material == nullptr)
    {
        using MU = Material::Uniform;
        std::shared_ptr<MU> muWS2VS = Renderer::Resources::Exist<MU>("mu_WS2VS")? Renderer::Resources::Get<MU>("mu_WS2VS") : Renderer::Resources::Create<MU>("mu_WS2VS")->SetType(MU::Type::Mat4x4);
        std::shared_ptr<MU> muVS2CS = Renderer::Resources::Exist<MU>("mu_VS2CS")? Renderer::Resources::Get<MU>("mu_VS2CS") : Renderer::Resources::Create<MU>("mu_VS2CS")->SetType(MU::Type::Mat4x4);

        m_material = Renderer::Resources::Create<Material>("mtr_Skybox");
        m_material->SetUniform("u_WS2VS", muWS2VS);
        m_material->SetUniform("u_VS2CS", muVS2CS);
        m_material->SetTexture("u_Skybox", m_cubemap);
    }
}

void RESkybox::_PrepareShader()
{
    if(m_shader == nullptr)
    {
        m_shader = Renderer::Resources::Create<Shader>("shader_Skybox")->LoadFromFile("/home/garra/study/dnn/assets/shader/Skybox_Cubemap.glsl");
    }
}

bool RESkybox::OnImgui()
{
    bool bChanged = false;
    if(ImGui::CollapsingHeader("SKYBOX"))
    {
        ImGui::Indent();
        if(ImGui::Button("SkyboxImage"))
        {
            const char* filename = osdialog_file(OSDIALOG_OPEN, "/home/garra/study/dnn/assets/texture/skybox", nullptr, nullptr);
            if(filename)
            {
                bChanged = _Reload(filename);
                delete[] filename;
            }
        }
        ImGui::SameLine();
        ImGui::Text("%s",  m_cubemap->GetImagePath().c_str());
        ImGui::Unindent();
    }
    return bChanged;
}

bool RESkybox::_Reload(const char* filename)
{
    if(m_cubemap->GetImagePath() == filename)
    {
        return false;
    }

    if(stbi_is_hdr(filename))
    {
        _GenCubemapFromEquirectangle(filename);
    }
    else
    {
        m_cubemap->Reload(filename, false);
    }

    return true;
}

void RESkybox::_GenCubemapFromEquirectangle(const char* filename)
{
    static bool bFirst = true;
    const unsigned int w = 2048;
    const unsigned int h = 2048;

    m_cubemap->SetImagePath(filename);
    m_cubemap->Set(w, h, Texture::Format::RGB16F);

    if(bFirst)
    {
        Renderer::Resources::Create<Viewport>("vp_CubemapFromEquirectangle")->SetRange(0, 0, w, h);

        std::shared_ptr<FrameBuffer> fbCapture = Renderer::Resources::Create<FrameBuffer>("fb_CubemapFromEquirectangle")->Set(w, h);
        fbCapture->AddRenderBuffer("rb_DepthStencil", RenderBuffer::Format::DEPTH_COMPONENT24);
        fbCapture->AddCubemapBuffer("cmb_CubemapFromEquirectangle", m_cubemap);

        using MU = Material::Uniform;
        glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
        std::shared_ptr<MU> muProjection = Renderer::Resources::Exist<MU>("mu_Projection")? Renderer::Resources::Get<MU>("mu_Projection") : Renderer::Resources::Create<MU>("mu_Projection")->Set(MU::Type::Mat4x4, 1, glm::value_ptr(captureProjection));
        std::shared_ptr<MU> muView = Renderer::Resources::Exist<MU>("mu_View")? Renderer::Resources::Get<MU>("mu_View") : Renderer::Resources::Create<MU>("mu_View")->SetType(MU::Type::Mat4x4);
        std::shared_ptr<Texture> texEquirectangle = Renderer::Resources::Create<Texture2D>("t2d_Skybox")->Load(filename, false);
        std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("mtr_CubemapFromEquirectangle");
        mtr->SetUniform("u_VS2CS", muProjection);
        mtr->SetUniform("u_WS2VS", muView);
        mtr->SetTexture("u_EquirectangularMap", texEquirectangle);

        std::shared_ptr<Shader> shader = Renderer::Resources::Create<Shader>("shader_CubemapFromEquirectangle")->Define("SEPERATEVP")->LoadFromFile("/home/garra/study/dnn/assets/shader/EquirectanglularToCubemap.glsl");
        Renderer::Resources::Create<Renderer::Element>("ele_CubemapFromEquirectangle")->Set(Renderer::Resources::Get<Elsa::Mesh>("mesh_Skybox"), mtr, shader);
        bFirst = false;
    }
    else
    {
        Renderer::Resources::Get<Texture2D>("t2d_Skybox")->Reload(filename, false);
    }

    std::shared_ptr<Viewport> vpCapture = Renderer::Resources::Get<Viewport>("vp_CubemapFromEquirectangle");
    std::shared_ptr<FrameBuffer> fbCapture = Renderer::Resources::Get<FrameBuffer>("fb_CubemapFromEquirectangle");
    std::shared_ptr<Material::Uniform> muView = Renderer::Resources::Get<Material::Uniform>("mu_View");

    glm::mat4 captureViews[6] = 
    {
        glm::lookAt(glm::vec3(0), glm::vec3(+1, 0, 0), glm::vec3(0, -1, 0)), 
        glm::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0)), 
        glm::lookAt(glm::vec3(0), glm::vec3(0, +1, 0), glm::vec3(0, 0, +1)), 
        glm::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, -1)), 
        glm::lookAt(glm::vec3(0), glm::vec3(0, 0, +1), glm::vec3(0, -1, 0)), 
        glm::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, -1, 0)), 
    };

    for(int i=0; i<6; i++)
    {
        fbCapture->UseCubemapFace("cmb_CubemapFromEquirectangle", TextureCubemap::Face(i));
        muView->UpdateData(glm::value_ptr(captureViews[i]));
        Renderer::BeginScene(vpCapture, fbCapture);
        Renderer::Submit("ele_CubemapFromEquirectangle");
        Renderer::EndScene();
    }
}
//////////////////////////////////////////////
RECubebox::RECubebox(const std::string& name)
    : Renderer::Element(name)
{
    _Prepare();
}

void RECubebox::_PrepareMesh()
{
    float vertices[] =  // position normal
    {
        // front 
        -1, -1, +1,  0, 0, +1,   
        +1, -1, +1,  0, 0, +1, 
        +1, +1, +1,  0, 0, +1, 
        -1, +1, +1,  0, 0, +1, 
        // back              
        +1, -1, -1,  0, 0, -1, 
        -1, -1, -1,  0, 0, -1, 
        -1, +1, -1,  0, 0, -1, 
        +1, +1, -1,  0, 0, -1, 
        // left              
        -1, -1, -1,  -1, 0, 0, 
        -1, -1, +1,  -1, 0, 0, 
        -1, +1, +1,  -1, 0, 0, 
        -1, +1, -1,  -1, 0, 0, 
        // right             
        +1, -1, +1,  +1, 0, 0, 
        +1, -1, -1,  +1, 0, 0, 
        +1, +1, -1,  +1, 0, 0, 
        +1, +1, +1,  +1, 0, 0, 
        // up                
        -1, +1, +1,  0, +1, 0, 
        +1, +1, +1,  0, +1, 0, 
        +1, +1, -1,  0, +1, 0, 
        -1, +1, -1,  0, +1, 0, 
        // down              
        -1, -1, -1,  0, -1, 0, 
        +1, -1, -1,  0, -1, 0, 
        +1, -1, +1,  0, -1, 0, 
        -1, -1, +1,  0, -1, 0, 
    };

    unsigned char indices[] = 
    { 
        0, 1, 2, 
        0, 2, 3, 
        4, 5, 6, 
        4, 6, 7, 
        8, 9, 10, 
        8, 10, 11, 
        12, 13, 14, 
        12, 14, 15, 
        16, 17, 18, 
        16, 18, 19, 
        20, 21, 22, 
        20, 22, 23
    };

    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout({Buffer::Element::DataType::UChar});
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout({{Buffer::Element::DataType::Float3, "a_Position", false}, {Buffer::Element::DataType::Float3, "a_Normal", false}});
    m_mesh = Renderer::Resources::Create<Elsa::Mesh>("mesh_Cubebox")->Set(ib, {vb});
}

void RECubebox::_PrepareTexture()
{

}

void RECubebox::_PrepareMaterial()
{
    m_material = Renderer::Resources::Create<Material>("mtr_Cubebox");
    m_material->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("ub_Transform"));
    m_material->SetTexture("u_Skybox", Renderer::Resources::Get<TextureCubemap>("tcm_Skybox"));

    using MU = Material::Uniform ;
    glm::mat4 m2w = Transform::Create("tf_m2w")->SetTranslation(glm::vec3(0, 0, 4))->Get();
    std::shared_ptr<MU> muM2W = Renderer::Resources::Create<MU>("mu_MS2WS_cb")->Set(MU::Type::Mat4x4, 1, glm::value_ptr(m2w));
    std::shared_ptr<MU> muRefractiveIndex = Renderer::Resources::Create<MU>("RefractiveIndex")->Set(MU::Type::Float1);
    std::shared_ptr<MU> muCameraPosition = Renderer::Resources::Exist<MU>("mu_CameraPosition")? Renderer::Resources::Get<MU>("mu_CameraPosition") : Renderer::Resources::Create<MU>("mu_CameraPosition")->SetType(MU::Type::Float3);
    m_material->SetUniform("u_MS2WS", muM2W);
    m_material->SetUniform("u_Camera.Position", muCameraPosition);
    m_material->SetUniform("u_RefractiveIndex", muRefractiveIndex);
    m_refractiveIndex = reinterpret_cast<float*>(muRefractiveIndex->GetData());
    *m_refractiveIndex = 1.33;
}

void RECubebox::_PrepareShader()
{
    m_shader = Renderer::Resources::Create<Shader>("shader_Reflect")->LoadFromFile("/home/garra/study/dnn/assets/shader/Reflect.glsl");
}

bool RECubebox::OnImgui()
{
    if(ImGui::CollapsingHeader("CUBEBOX"))
    {
        ImGui::Indent();
        static int e = 0;
        ImGui::RadioButton("Reflect", &e, 0);
        ImGui::RadioButton("Refract", &e, 1);
        if(e == 0)
        {
            m_shader = Renderer::Resources::Get<Shader>("shader_Reflect");
        }   
        else
        {
            m_shader = Renderer::Resources::Exist<Shader>("shader_Refract")? Renderer::Resources::Get<Shader>("shader_Refract") : Renderer::Resources::Create<Shader>("shader_Refract")->LoadFromFile("/home/garra/study/dnn/assets/shader/Refract.glsl");
            ImGui::SameLine();
            ImGui::DragFloat("RefractiveIndex", m_refractiveIndex, 0.01f, 1.0f, 3.0f);
        }
        ImGui::Unindent();
    }
    return false;
}


//////////////////////////////////////////////
RECubeboxCross::RECubeboxCross(const std::string& name)
    : Renderer::Element(name)
{
    _Prepare();
}

void RECubeboxCross::_PrepareMesh()
{
    if(m_mesh != nullptr)
    {
        return;
    }

    float vertices[] =  // pos stq
    {
        -1, -1, -1,  -1, -1, -1, 
        +1, -1, -1,  +1, -1, -1, 
        +1, +1, -1,  +1, +1, -1, 
        -1, +1, -1,  -1, +1, -1, 
                                 
        -1, -1, +1,  -1, -1, +1, 
        +1, -1, +1,  +1, -1, +1, 
        +1, +1, +1,  +1, +1, +1, 
        -1, +1, +1,  -1, +1, +1, 
      
        -1, -3, +1,  -1, -1, -1, 
        +1, -3, +1,  +1, -1, -1, 
                
        -3, -1, +1,  -1, -1, -1, 
        +3, -1, +1,  +1, -1, -1, 
        +5, -1, +1,  -1, -1, -1, 
                
        -3, +1, +1,  -1, +1, -1, 
        +3, +1, +1,  +1, +1, -1, 
        +5, +1, +1,  -1, +1, -1, 
                
        -1, +3, +1,  -1, +1, -1,  
        +1, +3, +1,  +1, +1, -1,  
    };

    unsigned char indices[] = 
    {
        0, 1, 2, 
        0, 2, 3, 
        4, 0, 3, 
        4, 3, 7, 
        5, 4, 7, 
        5, 7, 6, 
        1, 5, 6, 
        1, 6, 2, 
        0, 4, 5, 
        0, 5, 1, 
        3, 2, 6, 
        3, 6, 7, 

        8, 9, 5, 
        8, 5, 4, 
        10, 4, 7, 
        10, 7, 13, 
        5, 11, 14, 
        5, 14, 6, 
        11, 12, 15, 
        11, 15, 14, 
        7, 6, 17, 
        7, 17, 16, 
    };

    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout({Buffer::Element::DataType::UChar});
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout({{Buffer::Element::DataType::Float3, "a_Position", false}, {Buffer::Element::DataType::Float3, "a_TexCoord", false}});
    m_mesh = Renderer::Resources::Create<Elsa::Mesh>("mesh_CubeboxCross")->Set(ib, {vb});
}

void RECubeboxCross::_PrepareTexture()
{
    if(!Renderer::Resources::Exist<TextureCubemap>("tcm_Skybox"))
    {
        Renderer::Resources::Create<TextureCubemap>("tcm_Skybox")->Load("/home/garra/study/dnn/assets/texture/skybox/sunny/right.jpg", false);
    }
}

void RECubeboxCross::_PrepareMaterial()
{
    if(m_material == nullptr)
    {
        int lod = 0;
        using MU = Material::Uniform;
        std::shared_ptr<MU> muLOD = Renderer::Resources::Exist<MU>("mu_LOD")? Renderer::Resources::Get<MU>("mu_LOD") : Renderer::Resources::Create<MU>("mu_LOD")->Set(MU::Type::Int1, 1, &lod);
        m_material = Renderer::Resources::Create<Material>("mtr_CubeboxCross");
        m_material->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("ub_Transform"));
        m_material->SetUniform("u_LOD", muLOD);
        m_material->SetTexture("u_Cubemap", Renderer::Resources::Get<TextureCubemap>("tcm_Skybox"));
    }
}

void RECubeboxCross::_PrepareShader()
{
    if(m_shader == nullptr)
    {
        m_shader = Renderer::Resources::Create<Shader>("shader_CubeboxCross")->LoadFromFile("/home/garra/study/dnn/assets/shader/CubemapBox.glsl");
    }
}

bool RECubeboxCross::OnImgui()
{
    if(ImGui::CollapsingHeader("CUBEBOXCROSS"))
    {
        static int e = 0;
        ImGui::Indent();
        ImGui::RadioButton("Skybox", &e, 0);
        ImGui::RadioButton("DiffuseIrradiance", &e, 1);
        ImGui::RadioButton("Prefilter", &e, 2);

        std::shared_ptr<Material> mtrSkybox = Renderer::Resources::Get<Material>("mtr_Skybox");
        std::shared_ptr<Material> mtrCubeboxCross = Renderer::Resources::Get<Material>("mtr_CubeboxCross");
        if(e == 0)
        {
            std::shared_ptr<TextureCubemap> cubemap = Renderer::Resources::Get<TextureCubemap>("tcm_Skybox");
            mtrSkybox->SetTexture("u_Skybox", cubemap);
            mtrCubeboxCross->SetTexture("u_Cubemap", cubemap);
        }
        else if(e == 1)
        {
            std::shared_ptr<TextureCubemap> cubemap = Renderer::Resources::Get<TextureCubemap>("tcm_Irradiance");
            mtrSkybox->SetTexture("u_Skybox", cubemap);
            mtrCubeboxCross->SetTexture("u_Cubemap", cubemap);
        }
        else if(e == 2)
        {
            std::shared_ptr<TextureCubemap> cubemap = Renderer::Resources::Get<TextureCubemap>("tcm_Prefilter");
            mtrSkybox->SetTexture("u_Skybox", cubemap);
            mtrCubeboxCross->SetTexture("u_Cubemap", cubemap);
            static int lod = 0;
            ImGui::SameLine();
            ImGui::RadioButton("0", &lod, 0);
            ImGui::SameLine();
            ImGui::RadioButton("1", &lod, 1);
            ImGui::SameLine();
            ImGui::RadioButton("2", &lod, 2);
            ImGui::SameLine();
            ImGui::RadioButton("3", &lod, 3);
            ImGui::SameLine();
            ImGui::RadioButton("4", &lod, 4);
            Renderer::Resources::Get<Material::Uniform>("mu_LOD")->UpdateData(&lod);
        }
        ImGui::Unindent();
    }

    return false;
}

//////////////////////////////////////////////
RESpheres::RESpheres(const std::string& name)
    : Renderer::Element(name)
{
    _Prepare();
}

std::shared_ptr<Renderer::Element> RESpheres::Set(int row, int col, float spacing, float radius, int stacks, int sectors)
{
    m_row = row;
    m_col = col;
    m_spacing = spacing;
    m_radius = radius;
    m_stacks = stacks;
    m_sectors = sectors;
    m_nInstance = m_row*m_col;
    _Prepare();
    return shared_from_this();
}

void RESpheres::_PrepareMesh()
{
    struct VertexAttribute
    {
        glm::vec3 pos;
        glm::vec3 nor;
        glm::vec2 uv;
    };

    std::vector<VertexAttribute> vertices;
    std::vector<glm::i16vec3> triangles;
    const float PI = 3.14159265;
    float x, y, z, xz;
    float u, v;
    for(int i=0; i<=m_stacks; i++)
    {
        v = i/float(m_stacks);
        y = cosf(v*PI);
        xz = sinf(v*PI);
        for(int j=0; j<=m_sectors; j++)
        {
            u = j/float(m_sectors);
            x = xz*cosf(u*2*PI);
            z = xz*sinf(u*2*PI);
            vertices.push_back({ {m_radius*x, m_radius*y, m_radius*z}, {x, y, z}, {u, v} });
        }
    }

    int k1, k2;
    for(int i=0; i<m_stacks; i++)
    {
        k1 = i*(m_sectors+1);
        k2 = k1+m_sectors+1;
        for(int j=0; j<m_sectors; j++, k1++, k2++)
        {
            if(i != 0)
            {
                triangles.push_back({k1, k2, k1+1});
            }
            if(i != m_stacks-1)
            {
                triangles.push_back({k1+1, k2, k2+1});
            }
        }
    }

    std::shared_ptr<Transform> tf = Transform::Create("temp");
    glm::vec3 translation = glm::vec3(-2);
    struct InstanceAttribute
    {
        glm::mat4 m2w;
        glm::vec3 albedo;
        float metallic;
        float roughness;
        float ao;
    } 
    instances[m_nInstance];

    int k = 0;
    for(int i=0; i<m_row; i++)
    {
        translation.y = (i-m_row/2)*m_spacing;
        for(int j=0; j<m_col; j++)
        {
            translation.x = (j-m_col/2)*m_spacing;
            instances[k].m2w = tf->SetTranslation(translation)->Get();
//             instances[k].albedo = { rand()%256/255.0f, rand()%256/255.0f, rand()%256/255.0f };
            instances[k].albedo = {1, 0, 0};
            instances[k].metallic = i/float(m_row);
            instances[k].roughness = std::clamp(j/float(m_col), 0.05f, 1.0f);
            instances[k].ao = rand()%101/200.0f+0.5f;
            k++;
        }
    }

    Buffer::Layout layoutVertex   = { {Buffer::Element::DataType::Float3, "a_Position", false}, 
                                      {Buffer::Element::DataType::Float3, "a_Normal",   false}, 
                                      {Buffer::Element::DataType::Float2, "a_TexCoord", false} };
    Buffer::Layout layoutIndex    = { {Buffer::Element::DataType::UShort} };
    Buffer::Layout layoutInstance = { {Buffer::Element::DataType::Mat4,   "a_MS2WS",     false, 1}, 
                                      {Buffer::Element::DataType::Float3, "a_Albedo",    false, 1}, 
                                      {Buffer::Element::DataType::Float,  "a_Metallic",  false, 1}, 
                                      {Buffer::Element::DataType::Float,  "a_Roughness", false, 1}, 
                                      {Buffer::Element::DataType::Float,  "a_Ao",        false, 1} };
    
    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(vertices.size()*sizeof(VertexAttribute), &vertices[0])->SetLayout(layoutVertex);
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(triangles.size()*sizeof(glm::i16vec3), &triangles[0])->SetLayout(layoutIndex);
    std::shared_ptr<Buffer> instanceBuffer = Buffer::CreateVertex(m_nInstance*sizeof(InstanceAttribute), instances)->SetLayout(layoutInstance);
    if(m_mesh == nullptr)
    {
        m_mesh = Renderer::Resources::Create<Elsa::Mesh>("mesh_Spheres");
    }
    m_mesh->Set(indexBuffer, {vertexBuffer, instanceBuffer});
}


void RESpheres::_PrepareTexture()
{
    if(m_material == nullptr)
    {
        m_materialSource.NormalMap     = Renderer::Resources::Create<Texture2D>("t2d_NormalMap_spheres")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_normal.png");
        m_materialSource.AlbedoMap     = Renderer::Resources::Create<Texture2D>("t2d_AlbedoMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_basecolor.png");
        m_materialSource.RoughnessMap  = Renderer::Resources::Create<Texture2D>("t2d_RoughnessMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_roughness.png");
        m_materialSource.MetallicMap   = Renderer::Resources::Create<Texture2D>("t2d_MetallicMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_metallic.png");
        m_materialSource.AoMap         = Renderer::Resources::Create<Texture2D>("t2d_AoMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/ao.png"); 
        m_materialSource.LUTofBRDF     = Renderer::Resources::Exist<Texture2D>("t2d_BRDF")? Renderer::Resources::Get<Texture2D>("t2d_BRDF") : Renderer::Resources::Create<Texture2D>("t2d_BRDF");
        m_materialSource.IrradianceMap = Renderer::Resources::Exist<TextureCubemap>("tcm_Irradiance")? Renderer::Resources::Get<TextureCubemap>("tcm_Irradiance") : Renderer::Resources::Create<TextureCubemap>("tcm_Irradiance");
        m_materialSource.PrefilterMap  = Renderer::Resources::Exist<TextureCubemap>("tcm_Prefilter" )? Renderer::Resources::Get<TextureCubemap>("tcm_Prefilter" ) : Renderer::Resources::Create<TextureCubemap>("tcm_Prefilter" );
    }
}

void RESpheres::_PrepareMaterial()
{
    if(m_material == nullptr)
    {
        using MU = Material::Uniform;
        std::shared_ptr<MU> muCameraPosition = Renderer::Resources::Exist<MU>("mu_CameraPosition")? Renderer::Resources::Get<MU>("mu_CameraPosition") : Renderer::Resources::Create<MU>("mu_CameraPosition")->SetType(MU::Type::Float3);
        m_material = Renderer::Resources::Create<Material>("mtr_PBR");
        m_material->SetUniform("u_Camera.Position", muCameraPosition);

        m_material->SetTexture("u_NormalMap",     m_materialSource.NormalMap);
        m_material->SetTexture("u_AlbedoMap",     m_materialSource.AlbedoMap);
        m_material->SetTexture("u_RoughnessMap",  m_materialSource.RoughnessMap);
        m_material->SetTexture("u_MetallicMap",   m_materialSource.MetallicMap);
        m_material->SetTexture("u_AoMap",         m_materialSource.AoMap);
        m_material->SetTexture("u_IrradianceMap", m_materialSource.IrradianceMap);
        m_material->SetTexture("u_PrefilterMap",  m_materialSource.PrefilterMap);
        m_material->SetTexture("u_LUTofBRDF",     m_materialSource.LUTofBRDF);

        m_material->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("ub_Transform"));
        m_material->SetUniformBuffer("Lights",    Renderer::Resources::Get<UniformBuffer>("ub_Lights"));
    }
}

void RESpheres::_PrepareShader()
{
    if(m_shader == nullptr)
    {
        m_shader = Renderer::Resources::Create<Shader>("shader_Spheres")->Define("NUM_OF_POINTLIGHTS 4")->LoadFromFile("/home/garra/study/dnn/assets/shader/PBR.glsl");
    }
}

bool RESpheres::OnImgui()
{
    bool bChanged = false;
    if(ImGui::CollapsingHeader("MATERIALOFSPHERES"))
    {
        ImGui::Indent();
        bChanged |= ImGui::SliderInt("NumOfLights", &m_materialSource.NumOfLights, 0, 4);
        bChanged |= ImGui::Checkbox("AlbedoMap", &m_materialSource.HasAlbedoMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.AlbedoMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.AlbedoMap);
        }
        bChanged |= ImGui::Checkbox("AoMap", &m_materialSource.HasAoMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.AoMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.AoMap);
        }
        bChanged |= ImGui::Checkbox("MetallicMap", &m_materialSource.HasMetallicMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.MetallicMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.MetallicMap);
        }
        bChanged |= ImGui::Checkbox("NormalMap", &m_materialSource.HasNormalMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.NormalMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.NormalMap);
        }
        bChanged |= ImGui::Checkbox("RoughnessMap", &m_materialSource.HasRoughnessMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.RoughnessMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.RoughnessMap);
        }

        bChanged |= ImGui::Checkbox("DiffuseIrradianceMap", &m_materialSource.HasDiffuseIrradianceMap);
        bChanged |= ImGui::Checkbox("SpecularIrradianceMap", &m_materialSource.HasSpecularIrradianceMap);

        if(bChanged)
        {
            int macros = _GetShaderMacros();
            std::string shaderName = std::to_string(macros)+std::to_string(m_materialSource.NumOfLights);
            if(Renderer::Resources::Exist<Shader>(shaderName))
            {
                m_shader= Renderer::Resources::Get<Shader>(shaderName);
            }
            else
            {
                m_shader= Renderer::Resources::Create<Shader>(shaderName)->Define("NUM_OF_POINTLIGHTS "+std::to_string(m_materialSource.NumOfLights))->Define(macros)->LoadFromFile("/home/garra/study/dnn/assets/shader/PBR.glsl");
            }
        }
        ImGui::Unindent();
    }

    return bChanged;
}

int RESpheres::_GetShaderMacros() const
{
    int macros = 0;

    if(m_materialSource.HasAoMap)
    {
        macros |= static_cast<int>(Shader::Macro::AO_MAP);
    }
    if(m_materialSource.HasAlbedoMap)
    {
        macros |= static_cast<int>(Shader::Macro::ALBEDO_MAP);
    }
    if(m_materialSource.HasNormalMap)
    {
        macros |= static_cast<int>(Shader::Macro::NORMAL_MAP);
    }
    if(m_materialSource.HasMetallicMap)
    {
        macros |= static_cast<int>(Shader::Macro::METALLIC_MAP);
    }
    if(m_materialSource.HasRoughnessMap)
    {
        macros |= static_cast<int>(Shader::Macro::ROUGHNESS_MAP);
    }
    if(m_materialSource.HasDiffuseIrradianceMap)
    {
        macros |= static_cast<int>(Shader::Macro::IRRADIANCE_DIFFUSE_MAP);
    }
    if(m_materialSource.HasSpecularIrradianceMap)
    {
        macros |= static_cast<int>(Shader::Macro::IRRADIANCE_SPECULAR_MAP);
    }
    return macros;
}

/////////////////////////////////////////////////////////////////////
std::shared_ptr<Renderer::Element> REContainers::Set(unsigned int nInstance, float radius, float offset)
{
    m_nInstance = nInstance ;
    m_radius = radius;
    m_offset = offset;
    _Prepare();
    return shared_from_this();
}

void REContainers::_PrepareMesh()
{
    float vertices[] =   // (a_PositionMS, a_NormalMS, a_TangentMS, a_TexCoord)
    {
        // front 
        -1, -1, +1,  0, 0, +1,  +1, 0, 0,  0, 0,   
        +1, -1, +1,  0, 0, +1,  +1, 0, 0,  1, 0, 
        +1, +1, +1,  0, 0, +1,  +1, 0, 0,  1, 1, 
        -1, +1, +1,  0, 0, +1,  +1, 0, 0,  0, 1, 
        // back                            
        +1, -1, -1,  0, 0, -1,  -1, 0, 0,  0, 0, 
        -1, -1, -1,  0, 0, -1,  -1, 0, 0,  1, 0, 
        -1, +1, -1,  0, 0, -1,  -1, 0, 0,  1, 1, 
        +1, +1, -1,  0, 0, -1,  -1, 0, 0,  0, 1, 
        // left                            
        -1, -1, -1,  -1, 0, 0,  0, 0, +1,  0, 0, 
        -1, -1, +1,  -1, 0, 0,  0, 0, +1,  1, 0, 
        -1, +1, +1,  -1, 0, 0,  0, 0, +1,  1, 1, 
        -1, +1, -1,  -1, 0, 0,  0, 0, +1,  0, 1, 
        // right                           
        +1, -1, +1,  +1, 0, 0,  0, 0, -1,  0, 0, 
        +1, -1, -1,  +1, 0, 0,  0, 0, -1,  1, 0, 
        +1, +1, -1,  +1, 0, 0,  0, 0, -1,  1, 1, 
        +1, +1, +1,  +1, 0, 0,  0, 0, -1,  0, 1, 
        // up                              
        -1, +1, +1,  0, +1, 0,  +1, 0, 0,  0, 0, 
        +1, +1, +1,  0, +1, 0,  +1, 0, 0,  1, 0, 
        +1, +1, -1,  0, +1, 0,  +1, 0, 0,  1, 1, 
        -1, +1, -1,  0, +1, 0,  +1, 0, 0,  0, 1, 
        // down                            
        -1, -1, -1,  0, -1, 0,  -1, 0, 0,  0, 0, 
        +1, -1, -1,  0, -1, 0,  -1, 0, 0,  1, 0, 
        +1, -1, +1,  0, -1, 0,  -1, 0, 0,  1, 1, 
        -1, -1, +1,  0, -1, 0,  -1, 0, 0,  0, 1, 
    };

    unsigned char indices[] = 
    { 
        0, 1, 2, 
        0, 2, 3, 
        4, 5, 6, 
        4, 6, 7, 
        8, 9, 10, 
        8, 10, 11, 
        12, 13, 14, 
        12, 14, 15, 
        16, 17, 18, 
        16, 18, 19, 
        20, 21, 22, 
        20, 22, 23
    };

    glm::mat4 matM2W[m_nInstance];
    srand(time(NULL));
    std::shared_ptr<Transform> tf = Transform::Create("temp");
    glm::vec3 translation = glm::vec3(0);
    glm::vec3 rotation = glm::vec3(0);
    glm::vec3 scale = glm::vec3(1);
    int k = 0;
    for(unsigned int i=0; i<m_nInstance; i++)
    {
        float angle = i*360.0f/m_nInstance;
        translation.x = std::sin(angle)*m_radius+((rand()%(int)(2*m_offset*100))/100.0f-m_offset);
        translation.y = 0.4f*((rand()%(int)(2*m_offset*100))/100.0f-m_offset);
        translation.z = std::cos(angle)*m_radius+((rand()%(int)(2*m_offset*100))/100.0f-m_offset);

//         rotation.x = rand()%360;
//         rotation.y = rand()%360;
//         rotation.z = rand()%360;

        scale.x = (rand()%20)/20.0f+0.05f;
        scale.z = scale.y = scale.x;
        
        matM2W[k++] = tf->Set(translation, rotation, scale)->Get();
    }

    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float3, "a_PositionMS", false}, 
                                    {Buffer::Element::DataType::Float3, "a_NormalMS", false},
                                    {Buffer::Element::DataType::Float3, "a_TangentMS", false},
                                    {Buffer::Element::DataType::Float2, "a_TexCoord", false}  };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    Buffer::Layout layoutInstance = { {Buffer::Element::DataType::Mat4, "a_MS2WS", false, 1} };

    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Buffer> instanceBuffer = Buffer::CreateVertex(m_nInstance*sizeof(glm::mat4), matM2W)->SetLayout(layoutInstance);
    m_mesh = Renderer::Resources::Create<Elsa::Mesh>("mesh_Containers")->Set(indexBuffer, {vertexBuffer, instanceBuffer});
}


void REContainers::_PrepareTexture()
{

}


void REContainers::_PrepareMaterial()
{
    using MU = Material::Uniform;
    std::shared_ptr<MU> muEmissiveIntensity   = Renderer::Resources::Create<MU>("mu_EmissiveIntensity"  )->Set(MU::Type::Float1);
    std::shared_ptr<MU> muShininess           = Renderer::Resources::Create<MU>("mu_Shininess"          )->Set(MU::Type::Float1);
    std::shared_ptr<MU> muDisplacementScale   = Renderer::Resources::Create<MU>("mu_DisplacementScale"  )->Set(MU::Type::Float1);
    std::shared_ptr<MU> muBloomThreshold      = Renderer::Resources::Create<MU>("mu_BloomThreshold"     )->Set(MU::Type::Float1);
    std::shared_ptr<MU> muDiffuseReflectance  = Renderer::Resources::Create<MU>("mu_DiffuseReflectance" )->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.8f)));
    std::shared_ptr<MU> muSpecularReflectance = Renderer::Resources::Create<MU>("mu_SpecularReflectance")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MU> muEmissiveColor       = Renderer::Resources::Create<MU>("mu_EmissiveColor"      )->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    std::shared_ptr<MU> muCameraPosition      = Renderer::Resources::Create<MU>("mu_CameraPosition"     )->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(2.0f)));
    std::shared_ptr<MU> muAmbientColor        = Renderer::Resources::Create<MU>("mu_AmbientColor"       )->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.3f)));
    m_materialSource.diffuseReflectance  = reinterpret_cast<glm::vec3*>(muDiffuseReflectance ->GetData());
    m_materialSource.specularReflectance = reinterpret_cast<glm::vec3*>(muSpecularReflectance->GetData());
    m_materialSource.emissiveColor       = reinterpret_cast<glm::vec3*>(muEmissiveColor      ->GetData());
    m_materialSource.ambientColor        = reinterpret_cast<glm::vec3*>(muAmbientColor       ->GetData());
    m_materialSource.emissiveIntensity   = reinterpret_cast<    float*>(muEmissiveIntensity  ->GetData());
    m_materialSource.shininess           = reinterpret_cast<    float*>(muShininess          ->GetData());
    m_materialSource.displacementScale   = reinterpret_cast<    float*>(muDisplacementScale  ->GetData());
    m_materialSource.bloomThreshold      = reinterpret_cast<float*>(muBloomThreshold->GetData());
    m_materialSource.diffuseMap          = Renderer::Resources::Create<Texture2D>("t2d_DiffuseMap"     )->Load("/home/garra/study/dnn/assets/texture/wood.png");
    m_materialSource.normalMap           = Renderer::Resources::Create<Texture2D>("t2d_NormalMap"      )->Load("/home/garra/study/dnn/assets/texture/toy_box_normal.png");
    m_materialSource.displacementMap     = Renderer::Resources::Create<Texture2D>("t2d_DisplacementMap")->Load("/home/garra/study/dnn/assets/texture/toy_box_disp.png");
    m_materialSource.specularMap         = Renderer::Resources::Create<Texture2D>("t2d_SpecularMap"    )->Load("/home/garra/study/dnn/assets/texture/lighting_maps_specular_color.png");
    m_materialSource.emissiveMap         = Renderer::Resources::Create<Texture2D>("t2d_EmissiveMap"    )->Load("/home/garra/study/dnn/assets/texture/matrix.jpg");
    *m_materialSource.emissiveIntensity  = 1.0f;
    *m_materialSource.shininess          = 32.0f;
    *m_materialSource.displacementScale  = 0.1f;
    *m_materialSource.bloomThreshold     = 1.0f;


    m_material = Renderer::Resources::Create<Material>("mtr_Containers");
    m_material->SetUniform("u_Material.DiffuseReflectance"  , muDiffuseReflectance);
    m_material->SetUniform("u_Material.SpecularReflectance" , muSpecularReflectance);
    m_material->SetUniform("u_Material.EmissiveColor"       , muEmissiveColor);
    m_material->SetUniform("u_Material.EmissiveIntensity"   , muEmissiveIntensity);
    m_material->SetUniform("u_Material.Shininess"           , muShininess);
    m_material->SetUniform("u_Material.DisplacementScale"   , muDisplacementScale);
    m_material->SetTexture("u_Material.DiffuseMap"          , m_materialSource.diffuseMap);
    m_material->SetTexture("u_Material.NormalMap"           , m_materialSource.normalMap);
    m_material->SetTexture("u_Material.SpecularMap"         , m_materialSource.specularMap);
    m_material->SetTexture("u_Material.EmissiveMap"         , m_materialSource.emissiveMap);
    m_material->SetTexture("u_Material.DisplacementMap"     , m_materialSource.displacementMap);
    m_material->SetUniform("u_Camera.PositionWS"            , muCameraPosition);
    m_material->SetUniform("u_AmbientColor"                 , muAmbientColor);
    m_material->SetUniform("u_BloomThreshold"               , muBloomThreshold);

    m_material->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("ub_Transform"));
    m_material->SetUniformBuffer("Light"    , Renderer::Resources::Get<UniformBuffer>("Light"       ));
}

void REContainers::_PrepareShader()
{
    int macros = _GetShaderMacros();
    m_shader = Renderer::Resources::Create<Shader>(std::to_string(macros)+"_INSTANCE")->Define(macros)->Define("INSTANCE")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
}

int REContainers::_GetShaderMacros() const
{
    int macros = 0;
    if(m_materialSource.hasDiffuseReflectance)
    {
        macros |= static_cast<int>(Shader::Macro::DIFFUSE_REFLECTANCE);
    }
    if(m_materialSource.hasSpecularReflectance)
    {
        macros |= static_cast<int>(Shader::Macro::SPECULAR_REFLECTANCE);
    }
    if(m_materialSource.hasEmissiveColor)
    {
        macros |= static_cast<int>(Shader::Macro::EMISSIVE_COLOR);
    }
    if(m_materialSource.hasDiffuseMap)
    {
        macros |= static_cast<int>(Shader::Macro::DIFFUSE_MAP);
    }
    if(m_materialSource.hasSpecularMap)
    {
        macros |= static_cast<int>(Shader::Macro::SPECULAR_MAP);
    }
    if(m_materialSource.hasEmissiveMap)
    {
        macros |= static_cast<int>(Shader::Macro::EMISSIVE_MAP);
    }
    if(m_materialSource.hasNormalMap)
    {
        macros |= static_cast<int>(Shader::Macro::NORMAL_MAP);
    }
    if(m_materialSource.hasDisplacementMap)
    {
        macros |= static_cast<int>(Shader::Macro::DISPLACEMENT_MAP);
    }

    return macros;
}

bool REContainers::OnImgui()
{
    bool bChanged = false;
    if(ImGui::CollapsingHeader("MATERIALOFCONTAINERS"))
    {
        ImGui::Indent();
        bChanged |= ImGui::Checkbox("DiffuseRelectance", &m_materialSource.hasDiffuseReflectance);
        ImGui::SameLine(200);
        ImGui::ColorEdit3("DiffuseRelectance", (float*)m_materialSource.diffuseReflectance, ImGuiColorEditFlags_NoInputs|ImGuiColorEditFlags_NoLabel);

        bChanged |= ImGui::Checkbox("SpecularReflectance", &m_materialSource.hasSpecularReflectance);
        ImGui::SameLine(200);
        ImGui::ColorEdit3("SpecularReflectance", (float*)m_materialSource.specularReflectance, ImGuiColorEditFlags_NoInputs|ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine(250);
        ImGui::SetNextItemWidth(64);
        ImGui::DragFloat("Shininess", m_materialSource.shininess, 2, 2, 512, "%.0f");

        bChanged |= ImGui::Checkbox("EmissiveColor", &m_materialSource.hasEmissiveColor);
        ImGui::SameLine(200);
        ImGui::ColorEdit3("EmissiveColor", (float*)m_materialSource.emissiveColor, ImGuiColorEditFlags_NoInputs|ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine(250);
        ImGui::SetNextItemWidth(64);
        ImGui::DragFloat("Intensity", m_materialSource.emissiveIntensity, 0.1, 0, 10);

        bChanged |= ImGui::Checkbox("DiffuseMap", &m_materialSource.hasDiffuseMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.diffuseMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.diffuseMap);
        }
        bChanged |= ImGui::Checkbox("SpecularMap", &m_materialSource.hasSpecularMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.specularMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.specularMap);
        }
        bChanged |= ImGui::Checkbox("EmissiveMap", &m_materialSource.hasEmissiveMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.emissiveMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.emissiveMap);
        }
        bChanged |= ImGui::Checkbox("NormalMap", &m_materialSource.hasNormalMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.normalMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.normalMap);
        }
        bChanged |= ImGui::Checkbox("DisplacementMap", &m_materialSource.hasDisplacementMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_materialSource.displacementMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_materialSource.displacementMap);
        }
        ImGui::SameLine(250);
        ImGui::SetNextItemWidth(64);
        ImGui::DragFloat("DisplacementScale", m_materialSource.displacementScale,  0.001,  0,  1);

        if(bChanged)
        {
            int shaderMacros = _GetShaderMacros();
            std::string shaderName = std::to_string(shaderMacros)+"_INSTANCE";
            m_shader = Renderer::Resources::Exist<Shader>(shaderName)? Renderer::Resources::Get<Shader>(shaderName) : Renderer::Resources::Create<Shader>(shaderName)->Define("INSTANCE")->Define(shaderMacros)->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
        }
        ImGui::Unindent();
    }
    return bChanged;
}


////////////////////////////////////////////////////////
void REGroundPlane::_PrepareMesh()
{
    float vertices[] = 
    {
        -1, -1, 
        +1, -1, 
        +1, +1, 
        -1, +1,
    };
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout({ {Buffer::Element::DataType::Float2, "a_Position", false} });
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout({ {Buffer::Element::DataType::UChar} });
    m_mesh= Renderer::Resources::Create<Elsa::Mesh>("mesh_BackgroundPlane")->Set(ib, {vb});
}

void REGroundPlane::_PrepareTexture()
{

}

void REGroundPlane::_PrepareMaterial()
{
    using MU = Material::Uniform;
    std::shared_ptr<MU> muNearCorners = Renderer::Resources::Create<MU>("mu_NearCorners")->Set(MU::Type::Float3, 4);
    std::shared_ptr<MU> muFarCorners = Renderer::Resources::Create<MU>("mu_FarCorners")->Set(MU::Type::Float3, 4);
    m_material = Renderer::Resources::Create<Material>("mtr_GroundPlane");
    m_material->SetUniform("u_NearCorners", Renderer::Resources::Get<MU>("mu_NearCorners"));
    m_material->SetUniform("u_FarCorners", Renderer::Resources::Get<MU>("mu_FarCorners"));
    m_material->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("ub_Transform"));
}

void REGroundPlane::_PrepareShader()
{
    m_shader = Renderer::Resources::Create<Shader>("shader_GroundPlane")->LoadFromFile("/home/garra/study/dnn/assets/shader/GroundPlane.glsl");
}

bool REGroundPlane::OnImgui()
{
    if(ImGui::CollapsingHeader("GROUNDPLANE"))
    {
        ImGui::Indent();
        ImGui::Checkbox("Show##GroundPlane", &m_bVisible);
        ImGui::Unindent();
    }
    return false;
}
