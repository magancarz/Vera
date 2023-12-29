#pragma once

#include "renderEngine/RayTracing/Shapes/Triangle.h"
#include "Models/Vertex.h"

struct ModelData
{
	std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texture_coords;
    std::vector<float> tangents;
    std::vector<float> bitangents;
    std::vector<unsigned int> indices;
    std::vector<TriangleData> triangles;
};
