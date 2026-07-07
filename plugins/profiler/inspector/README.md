# NCCL Inspector Plugin

The NCCL Inspector is a plugin for the NVIDIA Collective Communications Library (NCCL) that provides detailed, per-communicator, per-collective performance and metadata logging. It is designed to help users analyze and debug NCCL collective operations by generating structured JSON output for each operation.

## Related Documentation

- **[Performance Exporter](exporter/example/README.md)** - Tool for analyzing and visualizing NCCL performance data from inspector logs

## Folder Location

The Inspector plugin source is located in:

```
plugins/profiler/inspector/
```

## Building the Inspector Plugin

To build the Inspector plugin, run:

```bash
make
```

The build system will automatically detect CUDA and NCCL installations from your environment. If you need to specify custom paths, you can set `CUDA_HOME` and `NCCL_HOME` environment variables or pass them as make arguments.

### Build Options

The Makefile supports several build options:

- **DEBUG=1**: Enable debug build with additional debugging information
- **ASAN=1**: Enable Address Sanitizer for memory error detection
- **UBSAN=1**: Enable Undefined Behavior Sanitizer

Example debug build:
```bash
make DEBUG=1
```

### Build Output

The build process creates:
- `libnccl-profiler-inspector.so`: The main inspector plugin library
- `version.cc`: Auto-generated version information from git

## Using NCCL Inspector

### Key Differences from Normal NCCL Usage

The main difference between running NCCL with the Inspector plugin versus running NCCL normally is the addition of environment variables that enable detailed performance logging:

**Normal NCCL Run:**
```bash
# Standard NCCL execution
./your_nccl_application
```

**NCCL Inspector Run:**
```bash
# NCCL Inspector enabled execution
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
./your_nccl_application
```

### Required Environment Variables

- `NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so`
  Loads the Inspector plugin into NCCL.
- `NCCL_INSPECTOR_ENABLE=1`
  Enables the Inspector plugin.

### Optional Environment Variables

The variables below are read by the current (refactored) plugin. They are grouped by purpose.

**Event tracking**

- `NCCL_INSPECTOR_ENABLE_P2P=<0|1>` (default: `1`)
  Enables or disables point-to-point (Send/Recv) tracking.
- `NCCL_INSPECTOR_ENABLE_PROXY=<0|1>` (default: `1`)
  Enables or disables proxy event tracking (`ProxyOp`/`ProxyStep`/`ProxyCtrl`). Proxy events are written only when verbose output is also enabled with `NCCL_INSPECTOR_DUMP_VERBOSE=1`.
- `NCCL_INSPECTOR_PROXY_MIN_MSG_SIZE=<bytes>` (default: `16777216`)
  Minimum message size (bytes) for proxy event tracking. Smaller messages are not tracked.
- `NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=<0|1>` (default: `1`)
  When enabled (default), only events with GPU-based kernel timing (`kernel_gpu`) are recorded. Events that fall back to CPU-measured timing (`kernel_cpu` or `collective_cpu`) are silently discarded. Set to `0` to retain all events regardless of timing source.
- `NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES=<bytes>` (default: `8192`)
  Minimum collective/P2P message size (bytes) to be tracked by inspector.

**Dump thread and output**

- `NCCL_INSPECTOR_DUMP_THREAD_ENABLE=<0|1>` (default: `1`)
  Enables or disables the internal dump thread that writes output to disk.
- `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=<interval>` (default: `-1`)
  Sets the interval (in microseconds) for the dump thread to write output. A value of `-1` (default) dumps only at communicator teardown/finalization. A value of `0` enables continuous dumping (dumps as fast as possible). A positive value enables periodic dumps at the specified interval (e.g., `500` for every 500 µs).
- `NCCL_INSPECTOR_DUMP_DIR=<output_dir>`
  Sets the output directory for logs. If not set, defaults to `nccl-inspector-<slurm_job_id>` when running under SLURM, otherwise `nccl-inspector-unknown-jobid`.
- `NCCL_INSPECTOR_DUMP_VERBOSE=<0|1>` (default: `0`)
  Enables verbose output including event-trace timestamps/sequence numbers and the nested child-event tree (`KernelCh`, proxy events) under each collective/P2P record. Also required for standalone proxy event records to be written.

**Buffer sizing (advanced)**

- `NCCL_INSPECTOR_ASYNC_QUEUE_SIZE=<entries>` (default: `1048576`)
  Capacity of the async event queue used by the consumer thread (async mode is always enabled). Clamped to a minimum of `1024`.
- `NCCL_INSPECTOR_DUMP_COLL_RING_SIZE=<entries>` (default: `1024`)
  Per-communicator completed-collective ring buffer capacity.
- `NCCL_INSPECTOR_DUMP_P2P_RING_SIZE=<entries>` (default: `1024`)
  Per-communicator completed-P2P ring buffer capacity.
- `NCCL_INSPECTOR_DUMP_PROXY_RING_SIZE=<entries>` (default: `8192`)
  Per-communicator completed-proxy-event ring buffer capacity.

### Debugging

To see detailed Inspector plugin messages, use NCCL's debug subsystem filtering. The Inspector uses the `PROFILE` subsystem:

```bash
# Show only Inspector messages
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROFILE

# Show Inspector messages along with other subsystems
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,PROFILE

# Show all debug messages (including Inspector)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

Inspector messages will appear with your configured NCCL_DEBUG level and will show:
- Plugin initialization and configuration
- Dump thread status and intervals
- File creation and locations
- Error conditions and warnings

### Example Usage

**Single Node:**
```bash
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
./build/test/perf/all_reduce_perf -b 8 -e 16G -f 2 -g 8
```

**Multi-Node (SLURM):**
```bash
# Add these environment variables to your SLURM script
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_DUMP_DIR=/path/to/logs/${SLURM_JOB_ID}/

# Then run your normal NCCL application
srun your_nccl_application
```

## JSON Dump Files

The inspector writes one log file per process into the dump directory, named `<hostname>-pid<pid>.log`. Each line is a self-contained JSON object (newline-delimited JSON) describing a single completed operation or event. Every object begins with two common blocks:

- `header`: communicator context — `id` (communicator hash), `comm_name`, `rank`, `n_ranks`, `nnodes`.
- `metadata`: `inspector_output_format_version` (currently `v4.1`), `git_rev`, `rec_mechanism`, `dump_timestamp_us`, `hostname`, `pid`.

After the common blocks, each record carries exactly one payload key identifying its type:

- `coll_perf`: a completed collective (e.g., AllReduce). Fields include `coll`, `coll_sn`, `coll_msg_size_bytes`, `coll_exec_time_us`, `coll_timing_source`, `coll_algobw_gbs`, `coll_busbw_gbs`.
- `p2p_perf`: a completed point-to-point operation. Fields include `p2p`, `p2p_sn`, `p2p_peer`, `p2p_msg_size_bytes`, `p2p_exec_time_us`, `p2p_timing_source`, `p2p_algobw_gbs`, `p2p_busbw_gbs`.
- `proxy_op` / `proxy_step` / `proxy_ctrl`: standalone network proxy events. Written only when `NCCL_INSPECTOR_DUMP_VERBOSE=1` and `NCCL_INSPECTOR_ENABLE_PROXY=1`.
- Generic profiler events, one key each: `group`, `group_api`, `coll_api`, `p2p_api`, `kernel_launch`, `net_plugin`. These are always captured (independent of verbose mode) and carry `start_ts`, `stop_ts`, `duration_us`, `rank` plus type-specific fields (e.g., `coll_api` adds `func`, `count`, `datatype`, `graph_captured`, `root`; `group_api` adds `group_depth`, `graph_captured`; `net_plugin` adds `net_plugin_id`).

When `NCCL_INSPECTOR_DUMP_VERBOSE=1`, `coll_perf` and `p2p_perf` records additionally embed `event_trace_sn`, `event_trace_ts`, and a nested `events` array holding the operation's child `KernelCh` and proxy events.

## Output Example

Each output file contains JSON objects with the following structure:

```json
{
  "header": {
    "id": "0x7f8c496ae9f661",
    "rank": 2,
    "n_ranks": 8,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.1",
    "git_rev": "",
    "rec_mechanism": "profiler_plugin",
    "dump_timestamp_us": 1748030377748202,
    "hostname": "example-hostname",
    "pid": 1639453
  },
  "coll_perf": {
    "coll": "AllReduce",
    "coll_sn": 1407,
    "coll_msg_size_bytes": 17179869184,
    "coll_exec_time_us": 61974,
    "coll_algobw_gbs": 277.210914,
    "coll_busbw_gbs": 485.119099
  }
}
```

## Output Example Verbose

To enable verbose output with event trace information, set the `NCCL_INSPECTOR_DUMP_VERBOSE=1` environment variable:

```bash
export NCCL_INSPECTOR_DUMP_VERBOSE=1
```

This will include additional event trace information in the JSON output, showing the sequence of callbacks and timestamps for each individual event. When proxy tracking is enabled, child profiler events are written as a nested tree under their parent collective or P2P operation: `coll_perf/p2p_perf -> events[]`. `ProxyOp` and `KernelCh` are sibling child events, and `ProxyStep` events are nested under their parent `ProxyOp`.

```json
{
  "header": {
    "id": "0xe62dedaa97644a",
    "rank": 4,
    "n_ranks": 8,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.1",
    "git_rev": "9019a1912-dirty",
    "rec_mechanism": "nccl_profiler_interface",
    "dump_timestamp_us": 1752867229276385,
    "hostname": "example-hostname",
    "pid": 438776
  },
  "coll_perf": {
    "coll": "ReduceScatter",
    "coll_sn": 1231,
    "coll_msg_size_bytes": 2147483648,
    "coll_exec_time_us": 41057,
    "coll_timing_source": "kernel_gpu",
    "coll_algobw_gbs": 418.439467,
    "coll_busbw_gbs": 366.134533,
    "event_trace_sn": {
      "coll_start_sn": 1,
      "coll_stop_sn": 2
    },
    "event_trace_ts": {
      "coll_start_ts": 1752867229235059,
      "coll_stop_ts": 1752867229235064
    },
    "events": [
      {
        "event_type": "KernelCh",
        "channel_id": 0,
        "event_trace_sn": {
          "kernel_start_sn": 3,
          "kernel_stop_sn": 48,
          "kernel_record_sn": 47
        },
        "event_trace_ts": {
          "kernel_start_ts": 1752867229235181,
          "kernel_stop_ts": 1752867229275811,
          "kernel_record_ts": 1752867229275811
        }
      },
      {
        "event_type": "ProxyOp",
        "proxy_id": 17,
        "proxy_rank": 4,
        "proxy_pid": 438776,
        "proxy_start_ts": 1752867229235201,
        "proxy_stop_ts": 1752867229275804,
        "proxy_duration_us": 40603,
        "proxy_direction": "Send",
        "proxy_channel_id": 0,
        "proxy_peer": 5,
        "proxy_n_steps": 8,
        "proxy_chunk_size": 524288,
        "proxy_trans_size": 2147483648,
        "proxy_states": [
          {
            "state": "ProxyOpInProgress",
            "state_id": 19,
            "state_ts": 1752867229235220
          }
        ],
        "proxy_step_count": 1,
        "events": [
          {
            "event_type": "ProxyStep",
            "proxy_step": 0,
            "proxy_start_ts": 1752867229235300,
            "proxy_stop_ts": 1752867229275600,
            "proxy_duration_us": 40300,
            "proxy_trans_size": 268435456,
            "proxy_states": [
              {
                "state": "ProxyStepSendWait",
                "state_id": 9,
                "state_ts": 1752867229235400,
                "trans_size": 268435456
              }
            ]
          }
        ]
      }
    ]
  }
}
```

Multiple such JSON objects are written, one per collective operation per communicator.

## Output Directory

- By default, output directory is auto-generated based on:
  - `nccl-inspector-<jobid>` if `SLURM_JOBID` is set
  - `nccl-inspector-unknown-jobid` otherwise
- You can override this with the `NCCL_INSPECTOR_DUMP_DIR` environment variable.

## Output File Size Estimates

- File size **grows continuously** throughout the application lifetime
- Each completed operation adds a new JSON line to the log file
- File size is proportional to:
  - Total number of collective/P2P operations executed
  - Number of parallel/overlapping communicators the process (PID) participates in
  - Whether verbose mode is enabled (verbose records are significantly larger)
- Estimate: ~200-500 bytes per operation (non-verbose)
- Example: A workload with 1M collectives across 4 communicators ≈ 200-500 MB per process

## Additional Notes

- The plugin is compatible with standard NCCL workflows and can be used in both single-node and multi-node (SLURM) environments.
- For more details, see the source code and comments in `plugins/profiler/inspector/`.
