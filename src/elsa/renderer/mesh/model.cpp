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
void Model::_ProcessMesh(const aiScene* scene, const aiMesh* mesh, unsigned int nthMesh)
{
    double scale(10.0);
//     scene->mMetaData->Get("UnitScaleFactor", scale);

    std::string meshName = mesh->mName.C_Str();
    meshName += "_";
    meshName += std::to_string(nthMesh);
//     INFO("Model::_ProcessMesh: {}, numVertices = {},\tnumFaces = {},\tscale = {}", meshName, mesh->mNumVertices, mesh->mNumFaces, scale);
    
    std::shared_ptr<Elsa::Mesh> elsaMesh = Elsa::Mesh::Create(meshName);
    elsaMesh->SetVertexNumber(mesh->mNumVertices);

    
    for(unsigned int i=0; i<mesh->mNumVertices; i++)
    {
        elsaMesh->PushVertex(glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z)/float(scale));
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
    m_renderElements.push_back(ele);
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
            INFO("{}:{}", glm::to_string(v), glm::to_string(vp*glm::vec4(v, 1)));
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
