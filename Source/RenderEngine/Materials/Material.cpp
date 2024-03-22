#include "Material.h"

Material::Material(
        const std::unique_ptr<DescriptorSetLayout>& material_descriptor_set_layout,
        const std::unique_ptr<DescriptorPool>& global_descriptor_set_pool,
        std::vector<std::shared_ptr<Texture>> textures)
    : textures{std::move(textures)}
{
    createImageInfosFromTextures();
    createTextureDescriptorSets(material_descriptor_set_layout, global_descriptor_set_pool);
}

void Material::createImageInfosFromTextures()
{
    assert(textures.size() > 0 && "Should be at least 1 texture for the material!");
    image_infos.reserve(textures.size());
    for (const auto& texture : textures)
    {
        VkDescriptorImageInfo image_info{};
        image_info.sampler = texture->getSampler();
        image_info.imageView = texture->getImageView();
        image_info.imageLayout = texture->getImageLayout();
        image_infos.emplace_back(image_info);
    }
}

void Material::createTextureDescriptorSets(
        const std::unique_ptr<DescriptorSetLayout>& material_descriptor_set_layout,
        const std::unique_ptr<DescriptorPool>& global_descriptor_set_pool)
{
    texture_descriptor_sets.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    for (auto& texture_descriptor_set : texture_descriptor_sets)
    {
        auto descriptor_writer = DescriptorWriter(*material_descriptor_set_layout, *global_descriptor_set_pool);

        for (size_t i = 0; i < image_infos.size(); ++i)
        {
            descriptor_writer.writeImage(i, &image_infos[i]);
        }
        descriptor_writer.build(texture_descriptor_set);
    }
}

VkDescriptorSet& Material::getDescriptorSet(size_t frame_index)
{
    return texture_descriptor_sets[frame_index];
}