#include "BVHTreeTraverser.h"

#include "BVHTreeNode.h"

#include "RenderEngine/RayTracing/Shapes/Triangle.h"

__device__ HitRecord BVHTreeTraverser::checkIntersection(const Ray* ray) const
{
    int to_visit_offset = 0;
    int current_node_index = 0;
    int nodes_to_visit[64];
    float closest_hit = FLT_MAX;
    HitRecord hit_record{};
    while (true)
    {
        BVHTreeNode* node = &nodes[current_node_index];
        float bounds_t_min = 0;
        if (node->bounds.intersect(ray, ray->inv_dir, ray->is_dir_neg, &bounds_t_min) && bounds_t_min < closest_hit)
        {
            if (node->num_of_shapes > 0)
            {
                for (int i = 0; i < node->num_of_shapes; ++i)
                {
                    const HitRecord temp = shapes[node->primitives_offset + i]->checkRayIntersection(ray);
                    if (temp.did_hit_anything && temp.t < closest_hit)
                    {
                        closest_hit = temp.t;
                        hit_record = temp;
                    }
                }
                if (to_visit_offset == 0) break;
                current_node_index = nodes_to_visit[--to_visit_offset];
            }
            else
            {
                if (ray->is_dir_neg[node->split_axis])
                {
                    nodes_to_visit[to_visit_offset++] = current_node_index + 1;
                    current_node_index = node->second_child_offset;
                }
                else
                {
                    nodes_to_visit[to_visit_offset++] = node->second_child_offset;
                    current_node_index = current_node_index + 1;
                }
            }
        }
        else
        {
            if (to_visit_offset == 0) break;
            current_node_index = nodes_to_visit[--to_visit_offset];
        }
    }

    return hit_record;
}
