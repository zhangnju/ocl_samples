

#include <iostream>
#include <iomanip>
#include <string>
#include <cassert>

#include <CL/cl.h>

#include "basic.hpp"
#include "oclobject.hpp"
#include "multidevice.hpp"


using namespace std;


int main (int argc, const char** argv)
{
    string platform_subname="0";
    cl_device_type device_type=CL_DEVICE_TYPE_ALL;
    //Scenario scenario=SCENARIO_SYSTEM_LEVEL;
    Scenario scenario=SCENARIO_SHARED_CONTEXT;
    int instance_count = 2;
    int instance_index = 0;
    size_t work_size = 16*1024*1024;

    if(argc ==2 )
    {
        int scen_id= atoi(argv[1]);
        if (scen_id== 0)
           scenario=SCENARIO_SYSTEM_LEVEL;
        else if(scen_id ==2)
           scenario=SCENARIO_SHARED_CONTEXT;
        else
           scenario=SCENARIO_MULTI_CONTEXT;
    }

    cl_platform_id platform = selectPlatform(platform_subname);

    switch(scenario)
    {
        case SCENARIO_SYSTEM_LEVEL:
            cout << "Executing system-level scenario." << endl;
            system_level_scenario(platform, device_type, work_size, instance_count, instance_index);
            break;
        case SCENARIO_MULTI_CONTEXT:
            cout << "Executing multi-context scenario." << endl;
            multi_context_scenario(platform, device_type, work_size);
            break;
        case SCENARIO_SHARED_CONTEXT:
            cout << "Executing shared-context scenario." << endl;
            shared_context_scenario(platform, device_type, work_size);
            break;
    }

    
    return EXIT_SUCCESS;
}
