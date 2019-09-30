/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : logger.cpp
* author      : Garra
* time        : 2019-09-02 20:31:30
* description : 
*
============================================*/


#include "logger.h"
#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> Logger::s_coreLogger;
std::shared_ptr<spdlog::logger> Logger::s_clientLogger;

void Logger::Init()
{
    spdlog::set_pattern("%^[%T] %n: %v%$");
    s_coreLogger = spdlog::stdout_color_mt("CORE");
    s_coreLogger->set_level(spdlog::level::trace);
    s_clientLogger = spdlog::stdout_color_mt("APP");
    s_clientLogger->set_level(spdlog::level::trace);
}

void Logger::Close()
{
    spdlog::drop_all();
}
