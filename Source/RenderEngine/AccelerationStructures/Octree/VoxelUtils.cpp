#include "VoxelUtils.h"

#include "Assets/Mesh.h"

std::unordered_set<Voxel> VoxelUtils::voxelize(const Mesh& mesh)
{
    std::unordered_set<Voxel> voxels;

    auto start = std::chrono::high_resolution_clock::now();

    uint32_t max_triangles{0};
    for (auto& model : mesh.models)
    {
        ModelDescription model_description = model->getModelDescription();
        max_triangles += model_description.num_of_triangles;

        for (size_t i = 0; i < model_description.num_of_indices; i += 3)
        {
            const glm::vec3 first = model->vertices[model->indices[i + 0]].position;
            const glm::vec3 second = model->vertices[model->indices[i + 1]].position;
            const glm::vec3 third = model->vertices[model->indices[i + 2]].position;

            const glm::vec3 edge_xy = second - first;
            const glm::vec3 edge_xz = third - first;

            float max_edge_length = std::max(glm::length(edge_xy), glm::length(edge_xz));
            float step = Voxel::DEFAULT_VOXEL_SIZE / max_edge_length;

            for (float u = 0.f; u <= 1.f; u += step)
            {
                for (float v = 0.f; v < 1.0f - u; v += step)
                {
                    glm::vec3 position = first + u * edge_xy + v * edge_xz;
                    glm::vec3 voxel_grid_position
                    {
                        glm::round(position.x / Voxel::DEFAULT_VOXEL_SIZE) * Voxel::DEFAULT_VOXEL_SIZE,
                        glm::round(position.y / Voxel::DEFAULT_VOXEL_SIZE) * Voxel::DEFAULT_VOXEL_SIZE,
                        glm::round(position.z / Voxel::DEFAULT_VOXEL_SIZE) * Voxel::DEFAULT_VOXEL_SIZE,
                    };
                    voxels.insert({ voxel_grid_position.x, voxel_grid_position.y, voxel_grid_position.z });
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    LogSystem::log(
        LogSeverity::LOG,
        "Voxelized model with ",
        max_triangles,
        " triangles in ",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(),
        " millis.");

    return voxels;
}
