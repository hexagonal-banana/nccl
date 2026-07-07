/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef INSPECTOR_INSPECTOR_TRACE_H_
#define INSPECTOR_INSPECTOR_TRACE_H_

#include "inspector.h"

// Converts an inspector JSON log file (one record per line) into a
// Chrome Trace Event Format JSON file loadable in Perfetto UI or
// chrome://tracing. This is the C++ port of convert_to_chrome_trace.py.
//
// Each inspector process only converts the single log file it produced.
//
// logPath  - path to the inspector .log file to read.
// outPath  - path to write the Chrome Trace JSON to.
inspectorResult_t inspectorConvertLogToChromeTrace(const char* logPath,
                                                   const char* outPath);

#endif  // INSPECTOR_INSPECTOR_TRACE_H_
