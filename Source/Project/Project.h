#pragma once

#include <string>
#include <vector>

#include "TriangleMeshInfo.h"

struct ProjectInfo
{
	std::string project_name;
	std::vector<TriangleMeshInfo> objects_infos;
};

class ProjectUtils
{
public:
	static void saveProject(const ProjectInfo& project_info);
	static ProjectInfo loadProject(const std::string& project_name);

	inline const static std::string PROJECTS_DIRECTORY = "Projects/";
	inline const static std::string PROJECT_FILE_EXTENSION = ".vproj";

private:
	static void saveProjectMetadata(std::ofstream& file_stream, const ProjectInfo& project_info);
	static void saveObjectsInfos(std::ofstream& file_stream, const ProjectInfo& project_info);

	static void loadProjectMetadata(std::string previous_line, std::ifstream& file_stream, ProjectInfo& project_info);
	static void loadObjectsInfos(std::string previous_line, std::ifstream& file_stream, ProjectInfo& project_info);

	inline const static std::string PROJECT_METADATA_PREFIX = "pm";
	inline const static std::string OBJECTS_INFOS_PREFIX = "oi";
};