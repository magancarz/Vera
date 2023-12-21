#pragma once

#include "renderEngine/RayTracing/Shapes/Triangle.h"
#include "Models/Vertex.h"

struct ModelData
{
	std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texture_coords;
    std::vector<std::optional<Vertex>> vertices;
    std::vector<unsigned int> indices;
    std::vector<TriangleData> triangles;
};
