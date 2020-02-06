/*============================================
* Copyright(C)2020 Garra. All rights reserved.
* 
* file        : src/utils/profile.cpp
* author      : Garra
* time        : 2020-01-13 17:14:48
* description : 
*
============================================*/


#include "profile.h"
#include "algorithm"


Profile& Profile::Get()
{
    static Profile profile;
    return profile;
}

void Profile::Begin(const std::string& filepath)
{
    m_fout.open(filepath);
    m_fout << "{\"otherData\":{},\"traceEvents\":[";
    m_fout.flush();
}

void Profile::End()
{
    m_count = 0;
    m_fout << "]}";
    m_fout.close();
}

void Profile::Write(const std::string& name, uint32_t tid, long long beg, long long end)
{
    std::lock_guard<std::mutex> lock(m_mutexer);
    if(m_count++>0)
    {
        m_fout << ",";
    }

    std::string _name = name;
    std::replace(_name.begin(), _name.end(), '"', '\'');

    m_fout << "{";
    m_fout << "\"name\":\"" << _name << "\",";
    m_fout << "\"cat\":\"function\",";
    m_fout << "\"ph\":\"X\",";
    m_fout << "\"pid\":0,";
    m_fout << "\"tid\":" << tid << ",";
    m_fout << "\"ts\":" << beg << ",";
    m_fout << "\"dur\":" << end-beg;
    m_fout << "}";
}
