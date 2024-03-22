#pragma once

#include <memory>

#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"

class Material
{
public:
    Material(
            const std::unique_ptr<DescriptorSetLayout>& material_descriptor_set_layout,
            const std::unique_ptr<DescriptorPool>& global_descriptor_set_pool,
            std::vector<std::shared_ptr<Texture>> textures);

    virtual ~Material() = default;

    [[nodiscard]] VkDescriptorSet& getDescriptorSet(size_t frame_index);

private:
    void createImageInfosFromTextures();
    void createTextureDescriptorSets(
            const std::unique_ptr<DescriptorSetLayout>& material_descriptor_set_layout,
            const std::unique_ptr<DescriptorPool>& global_descriptor_set_pool);

    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<VkDescriptorImageInfo> image_infos;
    std::vector<VkDescriptorSet> texture_descriptor_sets;
};
