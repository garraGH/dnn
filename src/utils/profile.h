/*============================================
* Copyright(C)2020 Garra. All rights reserved.
* 
* file        : src/utils/profile.h
* author      : Garra
* time        : 2020-01-13 17:15:35
* description : 
*
============================================*/


#pragma once
#include <string>
#include <fstream>
#include <mutex>

class Profile
{
public:
    void Begin(const std::string& filepath = "profile_results.json");
    void End();
    void Write(const std::string& name, uint32_t tid, long long beg, long long end);
    static Profile& Get();


private:
    int m_count = 0;
    std::mutex m_mutexer;
    std::ofstream m_fout;
};
