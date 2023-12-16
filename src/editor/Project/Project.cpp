#include "Project.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "Objects/ObjectInfo.h"

void ProjectUtils::saveProject(const ProjectInfo& project_info)
{
	std::ofstream file_stream(PROJECTS_DIRECTORY + project_info.project_name + PROJECT_FILE_EXTENSION);
	if (file_stream.is_open())
	{
		saveProjectMetadata(file_stream, project_info);
		saveObjectsInfos(file_stream, project_info);

		file_stream.close();
	}
}

void ProjectUtils::saveProjectMetadata(std::ofstream& file_stream, const ProjectInfo& project_info)
{
	file_stream << PROJECT_METADATA_PREFIX << " " << project_info.project_name << std::endl;
}

void ProjectUtils::saveObjectsInfos(std::ofstream& file_stream, const ProjectInfo& project_info)
{
	for (const auto& object_info : project_info.objects_infos)
	{
		file_stream << OBJECTS_INFOS_PREFIX << " " << object_info.object_name << " ";
		file_stream << object_info.model_name << " ";
		file_stream << object_info.material_name << " ";
		file_stream << object_info.position.x << " " << object_info.position.y << " " << object_info.position.z << " ";
		file_stream << object_info.rotation.x << " " << object_info.rotation.y << " " << object_info.rotation.z << " ";
		file_stream << object_info.scale << std::endl;
	}
}

ProjectInfo ProjectUtils::loadProject(const std::string& project_name)
{
	std::ifstream file_stream(PROJECTS_DIRECTORY + project_name + PROJECT_FILE_EXTENSION);
	if (file_stream.is_open())
	{
		ProjectInfo project_info{};

		loadProjectMetadata("", file_stream, project_info);

		file_stream.close();
		return project_info;
	}

	std::cout << "Couldn't find project named " << project_name << ".\n";
	return {};
}

void ProjectUtils::loadProjectMetadata(std::string previous_line, std::ifstream& file_stream, ProjectInfo& project_info)
{
	while (!previous_line.empty() || getline(file_stream, previous_line))
	{
		std::stringstream iss(previous_line);
		std::string prefix, name;
		iss >> prefix;
		if (!prefix.ends_with(PROJECT_METADATA_PREFIX))
		{
			return loadObjectsInfos(previous_line, file_stream, project_info);
		}

		iss >> name;
		project_info.project_name = name;
		previous_line = "";
	}
}

void ProjectUtils::loadObjectsInfos(std::string previous_line, std::ifstream& file_stream, ProjectInfo& project_info)
{
	std::vector<ObjectInfo> objects_infos;

	while (!previous_line.empty() || getline(file_stream, previous_line))
	{
		std::stringstream iss(previous_line);
		std::string prefix;
		iss >> prefix;
		if (!prefix.ends_with(OBJECTS_INFOS_PREFIX))
		{
			return;
		}

		std::string object_name, model_name, material_name, position_x, position_y, position_z,
			rotation_x, rotation_y, rotation_z, scale;
		iss >> object_name;
		iss >> model_name;
		iss >> material_name;
		iss >> position_x;
		iss >> position_y;
		iss >> position_z;
		glm::vec3 position{std::stof(position_x), std::stof(position_y), std::stof(position_z)};
		iss >> rotation_x;
		iss >> rotation_y;
		iss >> rotation_z;
		glm::vec3 rotation{std::stof(rotation_x), std::stof(rotation_y), std::stof(rotation_z)};
		iss >> scale;
		const float fscale = std::stof(scale);
		objects_infos.push_back(ObjectInfo{ object_name, model_name, material_name, position, rotation, fscale });
		previous_line = "";
	}

	project_info.objects_infos = objects_infos;
}
