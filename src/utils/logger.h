/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/opengl/logger.h
* author      : Garra
* time        : 2019-09-02 20:27:52
* description : 
*
============================================*/

#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"


#pragma once
class Logger
{
public:
    static void Init();
    static void Close();

    inline static std::shared_ptr<spdlog::logger>& getCoreLogger() { return s_coreLogger; };
    inline static std::shared_ptr<spdlog::logger>& getClientLogger() { return s_clientLogger; };

private:
    static std::shared_ptr<spdlog::logger> s_coreLogger;
    static std::shared_ptr<spdlog::logger> s_clientLogger;

};

// #ifdef LOG_TRACE
//     #define CORE_TRACE(...) Logger::getCoreLogger()->trace(__VA_ARGS__);
//     #define TRACE(...) Logger::getClientLogger()->trace(__VA_ARGS__);
// #else
//     #define CORE_TRACE(...)
//     #define TRACE(...)
// #endif
// 
// #ifdef LOG_INFO
//     #define CORE_INFO(...)  Logger::getCoreLogger()->info(__VA_ARGS__);
//     #define INFO(...)  Logger::getClientLogger()->info(__VA_ARGS__);
// #else
//     #define CORE_INFO(...)
//     #define INFO(...)
// #endif
// 
// #ifdef LOG_WARN
//     #define CORE_WARN(...)  Logger::getCoreLogger()->warn(__VA_ARGS__);
//     #define WARN(...)  Logger::getClientLogger()->warn(__VA_ARGS__);
// #else
//     #define CORE_WARN(...)
//     #define WARN(...)
// #endif

#define ENABLE_LOG

#ifdef ENABLE_LOG

#define     CORE_WARN(fmt, ...)     Logger::getCoreLogger()->warn    ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define     CORE_INFO(fmt, ...)     Logger::getCoreLogger()->info    ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define    CORE_TRACE(fmt, ...)     Logger::getCoreLogger()->trace   ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define    CORE_ERROR(fmt, ...)     Logger::getCoreLogger()->error   ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define CORE_CRITICAL(fmt, ...)     Logger::getCoreLogger()->critical("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);

#define     WARN(fmt, ...)     Logger::getClientLogger()->warn    ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define     INFO(fmt, ...)     Logger::getClientLogger()->info    ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define    TRACE(fmt, ...)     Logger::getClientLogger()->trace   ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define    ERROR(fmt, ...)     Logger::getClientLogger()->error   ("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define CRITICAL(fmt, ...)     Logger::getClientLogger()->critical("{}::{}({}): " fmt, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);

#else

#define     CORE_WARN(...)     
#define     CORE_INFO(...)     
#define    CORE_TRACE(...)    
#define    CORE_ERROR(...)    
#define CORE_CRITICAL(...) 

#define     WARN(...)     
#define     INFO(...)     
#define    TRACE(...)    
#define    ERROR(...)    
#define CRITICAL(...) 

#endif
