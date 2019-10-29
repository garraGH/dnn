/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/mesh/model.h
* author      : Garra
* time        : 2019-10-26 22:25:34
* description : 
*
============================================*/


#pragma once
#include <vector>
#include "../renderer.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "mesh.h"

class Model : public Asset, public std::enable_shared_from_this<Model>
{
public:
    Model(const std::string& name);

    std::shared_ptr<Model> LoadFromFile(const std::string& filepath);
    void Draw(const std::shared_ptr<Shader>& shader);
    void Export(const glm::mat4& vp);

    std::pair<glm::vec3, glm::vec3> GetAABB() const;
    static std::shared_ptr<Model> Create(const std::string& name);
protected:
    void _ProcessNode(aiNode* node, const aiScene* scene);
    void _ProcessMesh(const aiScene* scene, const aiMesh* mesh, unsigned int nthMesh);
    void _ProcessMaterial(aiMaterial* mtr, aiTextureType type, const std::string& typeName);

private:
    std::vector<std::shared_ptr<Elsa::Mesh>> m_meshes;
    std::vector<std::shared_ptr<Renderer::Element>> m_renderElements;
    std::vector<std::shared_ptr<Material>> m_materials;
    
};
