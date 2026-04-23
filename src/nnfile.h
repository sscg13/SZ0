#include <string>

#pragma once

std::string get_latest_onnx_file(const std::string &directory = ".");
void reload_network(const std::string &path);