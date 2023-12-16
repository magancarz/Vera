#include "OBJLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "models/AssetManager.h"

ModelData OBJLoader::loadModelDataFromOBJFileFormat(const std::string& file_name)
{
    const std::string file_path = locations::models_folder_location + file_name + locations::model_extension;

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./";

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

    std::vector<float> positions(attrib.vertices.size());
    std::vector<float> normals(attrib.vertices.size());
    std::vector<float> texture_coords(attrib.vertices.size() / 3 * 2);
    std::vector<std::optional<Vertex>> vertices(attrib.vertices.size() / 3);
    std::vector<unsigned int> indices;
    std::vector<TriangleData> triangles;

    size_t index_offset = 0;
    for (size_t face_index = 0; face_index < shapes[0].mesh.num_face_vertices.size(); ++face_index) {
        size_t num_of_face_vertices = shapes[0].mesh.num_face_vertices[face_index];

        bool calculate_normal{true};
        Vertex face_vertices[3];
        for (size_t face_vertex_index = 0; face_vertex_index < num_of_face_vertices; ++face_vertex_index) {
            tinyobj::index_t index = shapes[0].mesh.indices[index_offset + face_vertex_index];
            glm::vec3 position{
                    attrib.vertices[3*static_cast<size_t>(index.vertex_index) + 0],
                    attrib.vertices[3*static_cast<size_t>(index.vertex_index) + 1],
                    attrib.vertices[3*static_cast<size_t>(index.vertex_index) + 2]};

            glm::vec3 normal{0, 1, 0};
            if (index.normal_index >= 0)
            {
                normal = {
                    attrib.normals[3*static_cast<size_t>(index.normal_index) + 0],
                    attrib.normals[3*static_cast<size_t>(index.normal_index) + 1],
                    attrib.normals[3*static_cast<size_t>(index.normal_index) + 2]};
                calculate_normal = false;
            }

            glm::vec2 texture_coordinate{0.5f};
            if (index.texcoord_index >= 0)
            {
                texture_coordinate = {
                    attrib.texcoords[2*static_cast<size_t>(index.texcoord_index) + 0],
                    1.f - attrib.texcoords[2*static_cast<size_t>(index.texcoord_index) + 1]};
            }
            Vertex vertex{position, normal, texture_coordinate};
            face_vertices[face_vertex_index] = vertex;

            if (!vertices[index.vertex_index].has_value())
            {
                vertices[index.vertex_index] = vertex;

                positions[3 * index.vertex_index + 0] = face_vertices[face_vertex_index].position.x;
                positions[3 * index.vertex_index + 1] = face_vertices[face_vertex_index].position.y;
                positions[3 * index.vertex_index + 2] = face_vertices[face_vertex_index].position.z;

                normals[3 * index.vertex_index + 0] = face_vertices[face_vertex_index].normal.x;
                normals[3 * index.vertex_index + 1] = face_vertices[face_vertex_index].normal.y;
                normals[3 * index.vertex_index + 2] = face_vertices[face_vertex_index].normal.z;

                texture_coords[2 * index.vertex_index + 0] = face_vertices[face_vertex_index].texture_coordinate.x;
                texture_coords[2 * index.vertex_index + 1] = face_vertices[face_vertex_index].texture_coordinate.y;
            }

            indices.push_back(index.vertex_index);
        }

        triangles.push_back(
        {
            face_vertices[0],
            face_vertices[1],
            face_vertices[2],
        });

        if (calculate_normal)
        {
            glm::vec3 normal = glm::cross(face_vertices[1].position - face_vertices[0].position, face_vertices[2].position - face_vertices[0].position);
            face_vertices[0].normal = face_vertices[1].normal = face_vertices[2].normal = normal;
            for (size_t index = indices.size() - 3; index < indices.size(); ++index)
            {
                normals[3 * indices[index] + 0] = normal.x;
                normals[3 * indices[index] + 1] = normal.y;
                normals[3 * indices[index] + 2] = normal.z;
            }
        }

        index_offset += num_of_face_vertices;
    }

    return {positions, normals, texture_coords, vertices, indices, triangles};
}
