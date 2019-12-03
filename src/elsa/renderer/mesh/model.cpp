/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : model.cpp
* author      : Garra
* time        : 2019-10-26 22:25:32
* description : 
*
============================================*/


#include "model.h"
#include "../material/material.h"
#include "logger.h"
#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"

std::shared_ptr<Model> Model::Create(const std::string& name)
{
    return std::make_shared<Model>(name);
}

Model::Model(const std::string& name)
    : Asset( name )
{

}

std::shared_ptr<Model> Model::LoadFromFile(const std::string& filepath)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filepath, aiProcess_Triangulate|aiProcess_FlipUVs|aiProcess_OptimizeMeshes|aiProcess_GenBoundingBoxes|aiProcess_GlobalScale);
    if(!scene || !scene->mRootNode || scene->mFlags&AI_SCENE_FLAGS_INCOMPLETE)
    {
        ERROR("Model::LoadFromFile: %s, %s", filepath, importer.GetErrorString());
    }
    _ProcessNode(scene->mRootNode, scene);
    return shared_from_this();
}


void Model::_ProcessNode(aiNode* node, const aiScene* scene)
{
    for(unsigned int i=0; i<node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        _ProcessMesh(scene, mesh, i);
    }
    for(unsigned int i=0; i<node->mNumChildren; i++)
    {
        _ProcessNode(node->mChildren[i], scene);
    }
}

void Model::_DumpMeshInfo(const aiScene* scene, const aiMesh* mesh, const std::string& name)
{
    INFO("HasPositions              : {}, num: {}", mesh->HasPositions()? "Yes" : "No", mesh->mNumVertices);
    INFO("HasNormals                : {}", mesh->HasNormals()? "Yes" : "No");
    INFO("HasTangentsAndBitangents  : {}", mesh->HasTangentsAndBitangents()? "Yes" : "No");
    INFO("HasTextureCoords          : {}, num: {}", mesh->HasTextureCoords(0)? "Yes" : "No", mesh->mNumUVComponents[0]);
    INFO("HasVertexColors           : {}", mesh->HasVertexColors(0)? "Yes" : "No");
    INFO("HasFaces                  : {}, num: {}", mesh->HasFaces()? "Yes" : "No", mesh->mNumFaces);
    INFO("HasBones                  : {}, num: {}", mesh->HasBones()? "Yes" : "No", mesh->mNumBones);

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    INFO("Material: {}, {}", mesh->mMaterialIndex, material->GetName().C_Str());
    INFO("Properties: {}", material->mNumProperties);

    _ListTexture(material, aiTextureType_HEIGHT);
    _ListTexture(material, aiTextureType_AMBIENT);
    _ListTexture(material, aiTextureType_DIFFUSE);
    _ListTexture(material, aiTextureType_NORMALS);
    _ListTexture(material, aiTextureType_OPACITY);
    _ListTexture(material, aiTextureType_EMISSIVE);
    _ListTexture(material, aiTextureType_LIGHTMAP);
    _ListTexture(material, aiTextureType_SPECULAR);
    _ListTexture(material, aiTextureType_METALNESS);
    _ListTexture(material, aiTextureType_SHININESS);
    _ListTexture(material, aiTextureType_REFLECTION);
    _ListTexture(material, aiTextureType_DISPLACEMENT);
    _ListTexture(material, aiTextureType_BASE_COLOR);
    _ListTexture(material, aiTextureType_NORMAL_CAMERA);
    _ListTexture(material, aiTextureType_EMISSION_COLOR);
    _ListTexture(material, aiTextureType_DIFFUSE_ROUGHNESS);
}
void Model::_ProcessMesh(const aiScene* scene, const aiMesh* mesh, unsigned int nthMesh)
{
    double scale(1.0);
//     scene->mMetaData->Get("UnitScaleFactor", scale);

    std::string meshName = mesh->mName.C_Str();
    meshName += "_";
    meshName += std::to_string(nthMesh);
//     INFO("Model::_ProcessMesh: {}, numVertices = {},\tnumFaces = {},\tscale = {}", meshName, mesh->mNumVertices, mesh->mNumFaces, scale);
    
    _DumpMeshInfo(scene, mesh, meshName);
    std::shared_ptr<Elsa::Mesh> elsaMesh = Elsa::Mesh::Create(meshName);
    elsaMesh->SetVertexNumber(mesh->mNumVertices);

    
    for(unsigned int i=0; i<mesh->mNumVertices; i++)
    {
        Elsa::Mesh::Vertex vtx;
        vtx.pos = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z)/float(scale);
        vtx.nor = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        vtx.uv = mesh->mTextureCoords[0]? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : glm::vec2(0.0f);
//         INFO("{}, {}, {}", glm::to_string(vtx.pos), glm::to_string(vtx.nor), glm::to_string(vtx.uv));
        elsaMesh->PushVertex(vtx);
    }
    elsaMesh->SetIndexNumber(mesh->mNumFaces*3);
//     unsigned int k = 0;
    for(unsigned int i=0; i<mesh->mNumFaces; i++)
    {
//         INFO("FACE {}, mNumIndices: {}", i, mesh->mFaces[i].mNumIndices);
        for(unsigned int j=0; j<mesh->mFaces[i].mNumIndices; j++)
        {
//             TRACE("{}, {}", k++, mesh->mFaces[i].mIndices[j]);
            elsaMesh->PushIndex(mesh->mFaces[i].mIndices[j]);
        }
    }

    const aiVector3D& mMin = mesh->mAABB.mMin;
    const aiVector3D& mMax = mesh->mAABB.mMax;
    elsaMesh->SetAABB(mMin.x, mMin.y, mMin.z, mMax.x, mMax.y, mMax.z);
    elsaMesh->Build();
    m_meshes.push_back(elsaMesh);
    std::string eleName = "re_"+elsaMesh->GetName();
    std::shared_ptr<Renderer::Element> ele = Renderer::Resources::Create<Renderer::Element>(eleName);
    ele->SetMesh(elsaMesh);
    std::string textureBasePath = "/home/garra/study/dnn/assets/texture/";
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("mtr_"+m_name+"_"+std::to_string(nthMesh));
    using MU = Material::Uniform;
    mtr->SetUniform("u_Material.diffuseReflectance", Renderer::Resources::Get<MU>("MaterialDiffuseReflectance"));
    mtr->SetUniform("u_Material.specularReflectance", Renderer::Resources::Get<MU>("MaterialSpecularReflectance"));
    mtr->SetUniform("u_Material.emissiveColor", Renderer::Resources::Get<MU>("MaterialEmissiveColor"));
    mtr->SetUniform("u_Material.shininess", Renderer::Resources::Get<MU>("MaterialShininess"));

    // AmbientColor
    mtr->SetUniform("u_AmbientColor", Renderer::Resources::Get<MU>("AmbientColor"));

    if(mesh->mMaterialIndex >= 0)
    {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        aiTextureType type = aiTextureType_DIFFUSE;
        for(unsigned int i=0; i<material->GetTextureCount(type); i++)
        {
            aiString texturePath;
            material->GetTexture(type, 0, &texturePath);
            std::string temp = texturePath.C_Str();
            size_t pos = temp.find_last_of("/\\");
            std::string textureNameWithExtension = pos == std::string::npos? temp : temp.substr(pos+1);
            INFO("DiffuseTexture: {}", textureNameWithExtension);
            std::string textureNameWithoutExtension = textureNameWithExtension.substr(0, textureNameWithExtension.find('.'));
            std::shared_ptr<Texture> tex = Renderer::Resources::Create<Texture2D>(textureNameWithoutExtension)->Load(textureBasePath+textureNameWithExtension);
            mtr->SetTexture("u_Material.diffuseMap", tex);
        }
        type = aiTextureType_SPECULAR;
        for(unsigned int i=0; i<material->GetTextureCount(type); i++)
        {
            aiString texturePath;
            material->GetTexture(type, 0, &texturePath);
            std::string temp = texturePath.C_Str();
            size_t pos = temp.find_last_of("/\\");
            std::string textureNameWithExtension = pos == std::string::npos? temp : temp.substr(pos+1);
            INFO("SpecularTexture: {}", textureNameWithExtension);
            std::string textureNameWithoutExtension = textureNameWithExtension.substr(0, textureNameWithExtension.find('.'));
            std::shared_ptr<Texture> tex = Renderer::Resources::Create<Texture2D>(textureNameWithoutExtension)->Load(textureBasePath+textureNameWithExtension);
            mtr->SetTexture("u_Material.specularMap", tex);
        }
    }                                                        
    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    mtr->SetUniformBuffer("Light", Renderer::Resources::Get<UniformBuffer>("Light"));
    ele->SetMaterial(mtr);
    m_renderElements.push_back(ele);
}

const std::string Model::_TextureType(aiTextureType type) const
{
    switch(type)
    {
        case aiTextureType_HEIGHT: return "HEIGHT";
        case aiTextureType_AMBIENT: return "AMBIENT";
        case aiTextureType_DIFFUSE: return "DIFFUSE";
        case aiTextureType_NORMALS: return "NORMALS";
        case aiTextureType_OPACITY: return "OPACITY";
        case aiTextureType_EMISSIVE: return "EMISSIVE";
        case aiTextureType_LIGHTMAP: return "LIGHTMAP";
        case aiTextureType_SPECULAR: return "SPECULAR";
        case aiTextureType_METALNESS: return "METALNESS";
        case aiTextureType_SHININESS: return "SHININESS";
        case aiTextureType_REFLECTION: return "REFLECTION";
        case aiTextureType_DISPLACEMENT: return "DISPLACEMENT";
        case aiTextureType_BASE_COLOR: return "BASE_COLOR";
        case aiTextureType_NORMAL_CAMERA: return "NORMAL_CAMERA";
        case aiTextureType_EMISSION_COLOR: return "EMISSION_COLOR";
        case aiTextureType_AMBIENT_OCCLUSION: return "AMBIENT_OCCLUSION";
        case aiTextureType_DIFFUSE_ROUGHNESS: return "DIFFUSE_ROUGHNESS";
        default: return "UNKNOWN";
    }
}

void Model::_ListTexture(aiMaterial* material, aiTextureType type) const
{
    for(unsigned int i=0; i<material->GetTextureCount(type); i++)
    {
        aiString texturePath;
        material->GetTexture(type, 0, &texturePath);
        std::string temp = texturePath.C_Str();
        size_t pos = temp.find_last_of("/\\");
        std::string textureNameWithExtension = pos == std::string::npos? temp : temp.substr(pos+1);
        INFO("{}: {}", _TextureType(type), textureNameWithExtension);
        return;
    }

    INFO("{}: {}", _TextureType(type), "NULL");
}

void Model::_ProcessMaterial(aiMaterial* mtr, aiTextureType type, const std::string& typeName)
{

}

void Model::Draw(const std::shared_ptr<Shader>& shader)
{
    for(auto ele : m_renderElements)
    {
        Renderer::Submit(ele, shader);
    }
}

void Model::Export(const glm::mat4& vp)
{
    for(auto mesh : m_meshes)
    {
        auto vertices = mesh->GetVertices();
        for(auto v : vertices)
        {
            INFO("{}:{}", glm::to_string(v.pos), glm::to_string(vp*glm::vec4(v.pos, 1)));
            break;
        }
        break;
    }
}

std::pair<glm::vec3, glm::vec3> Model::GetAABB() const
{
    glm::vec3 mMin(1e10);
    glm::vec3 mMax(-1e10);
    for(auto mesh : m_meshes)    
    {
        auto [aMin, aMax] = mesh->GetAABB();
        mMin = glm::min(mMin, aMin);
        mMax = glm::max(mMax, aMax);
    }

    return {mMin, mMax};
}
