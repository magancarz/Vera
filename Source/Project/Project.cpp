#include "Project.h"
#include "Utils/PathBuilder.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

void ProjectUtils::saveProject(const ProjectInfo& project_info, const std::string& location)
{
    std::string file_location = PathBuilder(location).append(project_info.project_name + PROJECT_FILE_EXTENSION).build();
	std::ofstream file_stream(file_location);
	if (file_stream.is_open())
	{
		saveProjectMetadata(file_stream, project_info);
		saveObjectsInfos(file_stream, project_info);

		file_stream.close();

        std::cout << "Project named " << project_info.project_name << " saved successfully at location " << file_location << "\n";
        return;
	}

    std::cout << "Couldn't save project named " << project_info.project_name << " at location " << file_location << "!\n";
    std::cout << "Current working directory is " << std::filesystem::current_path() << "\n";
}

void ProjectUtils::saveProjectMetadata(std::ofstream& file_stream, const ProjectInfo& project_info)
{
	file_stream << PROJECT_METADATA_PREFIX << " " << project_info.project_name << std::endl;
}

void ProjectUtils::saveObjectsInfos(std::ofstream& file_stream, const ProjectInfo& project_info)
{
	for (const auto& object_info : project_info.objects_infos)
	{
		file_stream << OBJECTS_INFOS_PREFIX << " " << object_info.toString() << std::endl;
	}
}

ProjectInfo ProjectUtils::loadProject(const std::string& project_name, const std::string& location)
{
	std::ifstream file_stream(location + project_name + PROJECT_FILE_EXTENSION);
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
		if (!prefix.starts_with(PROJECT_METADATA_PREFIX))
		{
            loadObjectsInfos(previous_line, file_stream, project_info);
			return;
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
		if (!prefix.starts_with(OBJECTS_INFOS_PREFIX))
		{
            break;
		}

        std::string object_info_as_string;
        getline(iss, object_info_as_string);
		objects_infos.push_back(ObjectInfo::fromString(object_info_as_string));
        previous_line = "";
	}

    project_info.objects_infos = objects_infos;
}
