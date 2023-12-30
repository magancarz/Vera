#pragma once

#include <string>
#include <vector>

class Object;
struct TriangleMeshInfo;

struct ProjectInfo
{
	std::string project_name;
	std::vector<std::string> objects_infos;
	std::vector<std::string> lights_infos;
};

class ProjectUtils
{
public:
	static void saveProject(const ProjectInfo& project_info);
	static ProjectInfo loadProject(const std::string& project_name);

	inline const static std::string PROJECTS_DIRECTORY = "Projects/";
	inline const static std::string PROJECT_FILE_EXTENSION = ".tinproj";

private:
	static void saveProjectMetadata(std::ofstream& file_stream, const ProjectInfo& project_info);
	static void saveObjectsInfos(std::ofstream& file_stream, const ProjectInfo& project_info);
	static void saveLightsInfos(std::ofstream& file_stream, const ProjectInfo& project_info);

	static void loadProjectMetadata(std::string previous_line, std::ifstream& file_stream, ProjectInfo& project_info);
	static void loadObjectsInfos(std::string previous_line, std::ifstream& file_stream, ProjectInfo& project_info);
	static void loadLightsInfos(std::string previous_line, std::ifstream& file_stream, ProjectInfo& project_info);

	inline const static std::string PROJECT_METADATA_PREFIX = "pm";
	inline const static std::string OBJECTS_INFOS_PREFIX = "oi";
	inline const static std::string LIGHTS_INFOS_PREFIX = "li";
};