//===- dpctl_service.cpp - C API for service functions   -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header defines dpctl service functions.
///
//===----------------------------------------------------------------------===//

#include "dpctl_service.h"
#include "Config/dpctl_config.h"

#include "dpctl_string_utils.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#ifdef ENABLE_GLOG
#include <filesystem>
#include <glog/logging.h>
#endif

__dpctl_give const char *DPCTLService_GetDPCPPVersion(void)
{
    std::string version = DPCTL_DPCPP_VERSION;
    return dpctl::helper::cstring_from_string(version);
}

#ifdef ENABLE_GLOG

void DPCTLService_InitLogger(const char *app_name, const char *log_dir)
{
    google::InitGoogleLogging(app_name);
    google::InstallFailureSignalHandler();

    if (log_dir) {
        namespace fs = std::filesystem;
        const fs::path path(log_dir);
        std::error_code ec;

        if (fs::is_directory(path, ec)) {
            google::EnableLogCleaner(0);
            FLAGS_log_dir = log_dir;
        }
    }
    else {
        FLAGS_colorlogtostderr = true;
        FLAGS_stderrthreshold = google::FATAL;
        FLAGS_logtostderr = 1;
    }
}

void DPCTLService_ShutdownLogger(void)
{
    google::ShutdownGoogleLogging();
}

#else
void DPCTLService_InitLogger([[maybe_unused]] const char *app_name,
                             [[maybe_unused]] const char *log_dir){};

void DPCTLService_ShutdownLogger(void){};

#endif
