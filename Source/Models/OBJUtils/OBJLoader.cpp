#include "OBJLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "models/AssetManager.h"

namespace std
{
    template<> struct hash<Vertex>
    {
        size_t operator()(Vertex const& vertex) const
        {
            return ((hash<glm::vec3>()(vertex.position) ^
                     (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texture_coordinate) << 1);
        }
    };
}

ModelData OBJLoader::loadModelDataFromOBJFileFormat(const std::string& file_name)
{
    const std::string file_path = locations::models_folder_location + file_name + locations::model_extension;

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./";
    reader_config.triangulate = true;

    tinyobj::ObjReader obj_reader;

    if (!obj_reader.ParseFromFile(file_path, reader_config)) {
      if (!obj_reader.Error().empty()) {
          std::cerr << "TinyObjReader: " << obj_reader.Error();
      }
      exit(1);
    }

    if (!obj_reader.Warning().empty()) {
      std::cout << "TinyObjReader: " << obj_reader.Warning();
    }

    auto& attrib = obj_reader.GetAttrib();
    auto& shapes = obj_reader.GetShapes();

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texture_coords;
    std::vector<float> tangents;
    std::vector<float> bitangents;
    std::vector<unsigned int> indices;
    std::vector<TriangleData> triangles;

    std::unordered_map<Vertex, uint32_t> unique_vertices{};
    size_t index_offset = 0;
    for (size_t face_index = 0; face_index < shapes[0].mesh.num_face_vertices.size(); ++face_index) {
        size_t num_of_face_vertices = shapes[0].mesh.num_face_vertices[face_index];

        bool calculate_normal{false};
        Vertex face_vertices[3];
        for (size_t face_vertex_index = 0; face_vertex_index < num_of_face_vertices; ++face_vertex_index) {
            tinyobj::index_t index = shapes[0].mesh.indices[index_offset++];
            glm::vec3 position
            {
                attrib.vertices[3 * static_cast<size_t>(index.vertex_index) + 0],
                attrib.vertices[3 * static_cast<size_t>(index.vertex_index) + 1],
                attrib.vertices[3 * static_cast<size_t>(index.vertex_index) + 2]
            };

            glm::vec3 normal{0, 1, 0};
            if (!calculate_normal && index.normal_index >= 0)
            {
                normal =
                {
                    attrib.normals[3 * static_cast<size_t>(index.normal_index) + 0],
                    attrib.normals[3 * static_cast<size_t>(index.normal_index) + 1],
                    attrib.normals[3 * static_cast<size_t>(index.normal_index) + 2]
                };
            }
            else
            {
                calculate_normal = true;
            }

            glm::vec2 texture_coordinate{0.5f, 0.5f};
            if (index.texcoord_index >= 0)
            {
                texture_coordinate =
                {
                    attrib.texcoords[2 * static_cast<size_t>(index.texcoord_index) + 0],
                    1.0f - attrib.texcoords[2 * static_cast<size_t>(index.texcoord_index) + 1]
                };
            }
            Vertex vertex{position, normal, texture_coordinate};
            face_vertices[face_vertex_index] = vertex;
        }

        if (calculate_normal)
        {
            glm::vec3 normal = glm::normalize(glm::cross(face_vertices[1].position - face_vertices[0].position, face_vertices[2].position - face_vertices[0].position));
            face_vertices[0].normal = face_vertices[1].normal = face_vertices[2].normal = normal;
        }

        glm::vec3 edge1 = face_vertices[1].position - face_vertices[0].position;
        glm::vec3 edge2 = face_vertices[2].position - face_vertices[0].position;
        glm::vec2 delta_UV1 = face_vertices[1].texture_coordinate - face_vertices[0].texture_coordinate;
        glm::vec2 delta_UV2 = face_vertices[2].texture_coordinate - face_vertices[0].texture_coordinate;

        glm::vec3 tangent, bitangent;
        float f = 1.0f / (delta_UV1.x * delta_UV2.y - delta_UV2.x * delta_UV1.y);
        tangent.x = f * (delta_UV2.y * edge1.x - delta_UV1.y * edge2.x);
        tangent.y = f * (delta_UV2.y * edge1.y - delta_UV1.y * edge2.y);
        tangent.z = f * (delta_UV2.y * edge1.z - delta_UV1.y * edge2.z);
        tangent = glm::normalize(tangent);

        bitangent.x = f * (-delta_UV2.x * edge1.x + delta_UV1.x * edge2.x);
        bitangent.y = f * (-delta_UV2.x * edge1.y + delta_UV1.x * edge2.y);
        bitangent.z = f * (-delta_UV2.x * edge1.z + delta_UV1.x * edge2.z);
        bitangent = glm::normalize(bitangent);

        face_vertices[0].tangent = face_vertices[1].tangent = face_vertices[2].tangent = tangent;
        face_vertices[0].bitangent = face_vertices[1].bitangent = face_vertices[2].bitangent = bitangent;

        for (const Vertex& vertex : face_vertices)
        {
            if (unique_vertices.count(vertex) == 0)
            {
                unique_vertices[vertex] = static_cast<uint32_t>(positions.size() / 3);

                positions.push_back(vertex.position.x);
                positions.push_back(vertex.position.y);
                positions.push_back(vertex.position.z);

                normals.push_back(vertex.normal.x);
                normals.push_back(vertex.normal.y);
                normals.push_back(vertex.normal.z);

                texture_coords.push_back(vertex.texture_coordinate.x);
                texture_coords.push_back(vertex.texture_coordinate.y);

                tangents.push_back(vertex.tangent.x);
                tangents.push_back(vertex.tangent.y);
                tangents.push_back(vertex.tangent.z);

                bitangents.push_back(vertex.bitangent.x);
                bitangents.push_back(vertex.bitangent.y);
                bitangents.push_back(vertex.bitangent.z);
            }

            indices.push_back(unique_vertices[vertex]);
        }

        triangles.push_back(
        {
                face_vertices[0],
                face_vertices[1],
                face_vertices[2],
        });
    }

    return {positions, normals, texture_coords, tangents, bitangents, indices, triangles};
}
