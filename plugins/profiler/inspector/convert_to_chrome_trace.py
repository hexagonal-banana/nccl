#!/usr/bin/env python3
"""Convert NCCL Inspector profiler log files to Chrome Trace Event Format JSON.

Usage:
    python convert_to_chrome_trace.py <input_log_or_dir> [<input_log_or_dir> ...] [-o output.json]

Examples:
    # Single file
    python convert_to_chrome_trace.py results/dump/node-pid12345.log -o trace.json

    # All logs in a directory
    python convert_to_chrome_trace.py results/dump/ -o trace.json

    # Multiple files
    python convert_to_chrome_trace.py rank0.log rank1.log -o trace.json

Output is Chrome Trace JSON loadable in chrome://tracing or Perfetto UI.
"""

import argparse
import json
import sys
from pathlib import Path


def collect_log_files(paths):
    """Resolve input paths to a list of .log files."""
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.log")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"WARNING: skipping non-existent path: {p}", file=sys.stderr)
    return files


def parse_log_file(filepath):
    """Parse a log file, yielding one parsed JSON dict per line."""
    with open(filepath, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: {filepath}:{line_no}: JSON parse error: {e}", file=sys.stderr)


def make_pid_str(rank, hostname):
    """Build string pid like 'rank0@hostname'."""
    return f"rank{rank}@{hostname}"


def convert_proxy_ctrl_event(record, trace_events, pid_str):
    """Convert a standalone proxy_event (ProxyCtrl) to trace events."""
    proxy = record["proxy_event"]
    proxy_id = proxy["proxy_id"]

    start_ts = proxy["proxy_start_ts"]
    stop_ts = proxy["proxy_stop_ts"]
    dur = stop_ts - start_ts

    states = proxy.get("proxy_states", [])

    # Event name = first state name (e.g., "ProxyCtrlSleep", "ProxyCtrlIdle")
    name = states[0]["state"] if states else "ProxyCtrl"

    # Emit X event for the ProxyCtrl span
    trace_events.append({
        "name": name,
        "cat": "proxy_ctrl",
        "ph": "X",
        "ts": start_ts,
        "dur": dur,
        "pid": pid_str,
        "tid": "ProxyCtrl",
        "args": {
            "proxy_id": proxy_id,
            "duration_us": dur,
        },
    })

    # Emit state transition instants
    for state in states:
        trace_events.append({
            "name": state["state"],
            "cat": "state",
            "ph": "i",
            "s": "t",
            "ts": state["state_ts"],
            "pid": pid_str,
            "tid": "ProxyCtrl",
            "args": {
                "state_id": state["state_id"],
            },
        })


def convert_coll_perf(record, trace_events, pid_str):
    """Convert a coll_perf record to trace events."""
    header = record["header"]
    coll = record["coll_perf"]

    rank = header["rank"]
    coll_name = coll["coll"]
    coll_sn = coll["coll_sn"]
    msg_size = coll["coll_msg_size_bytes"]

    # --- Collective API span ---
    coll_start = coll["event_trace_ts"]["coll_start_ts"]
    coll_stop = coll["event_trace_ts"]["coll_stop_ts"]
    coll_dur = coll_stop - coll_start

    trace_events.append({
        "name": f"{coll_name} #{coll_sn}",
        "cat": "coll",
        "ph": "X",
        "ts": coll_start,
        "dur": coll_dur,
        "pid": pid_str,
        "tid": "Collective",
        "args": {
            "coll_sn": coll_sn,
            "msg_size_bytes": msg_size,
            "exec_time_us": coll["coll_exec_time_us"],
            "timing_source": coll.get("coll_timing_source", ""),
            "algobw_gbs": coll.get("coll_algobw_gbs"),
            "busbw_gbs": coll.get("coll_busbw_gbs"),
            "comm_id": header["id"],
            "rank": rank,
        },
    })

    # --- Process nested events ---
    events = coll.get("events", [])

    for ev in events:
        ev_type = ev["event_type"]

        if ev_type == "KernelCh":
            convert_kernel_ch(ev, coll_sn, pid_str, trace_events)
        elif ev_type == "ProxyOp":
            convert_proxy_op(ev, coll_sn, pid_str, trace_events)


def convert_kernel_ch(ev, coll_sn, pid_str, trace_events):
    """Convert a KernelCh event to a trace event on the single Kernel tid."""
    channel_id = ev["channel_id"]
    k_start = ev["event_trace_ts"]["kernel_start_ts"]
    k_stop = ev["event_trace_ts"]["kernel_stop_ts"]
    k_dur = k_stop - k_start

    trace_events.append({
        "name": f"KernelCh{channel_id}",
        "cat": "kernel",
        "ph": "X",
        "ts": k_start,
        "dur": k_dur,
        "pid": pid_str,
        "tid": "Kernel",
        "args": {
            "channel_id": channel_id,
            "coll_sn": coll_sn,
            "trace_sn": ev.get("event_trace_sn", {}),
        },
    })


def convert_proxy_op(ev, coll_sn, pid_str, trace_events):
    """Convert a ProxyOp event to trace events.

    Emits ProxyOp as an X event, and all state transitions (from both
    ProxyOp and nested ProxySteps) as instant events on the same tid.
    """
    proxy_id = ev["proxy_id"]
    direction = ev["proxy_direction"]  # "Send" or "Recv"
    channel_id = ev["proxy_channel_id"]
    peer = ev["proxy_peer"]

    op_start = ev["proxy_start_ts"]
    op_stop = ev["proxy_stop_ts"]
    op_dur = op_stop - op_start

    # tid: unique per channel + direction + peer
    tid = f"Proxy ch{channel_id} {direction}\u2192peer{peer}"

    trace_events.append({
        "name": f"ProxyOp {direction} ch{channel_id}",
        "cat": "proxy",
        "ph": "X",
        "ts": op_start,
        "dur": op_dur,
        "pid": pid_str,
        "tid": tid,
        "args": {
            "proxy_id": proxy_id,
            "direction": direction,
            "channel_id": channel_id,
            "peer": peer,
            "n_steps": ev.get("proxy_n_steps"),
            "chunk_size": ev.get("proxy_chunk_size"),
            "trans_size": ev.get("proxy_trans_size"),
            "duration_us": ev.get("proxy_duration_us"),
            "step_count": ev.get("proxy_step_count"),
            "coll_sn": coll_sn,
        },
    })

    # ProxyOp state instants
    for state in ev.get("proxy_states", []):
        trace_events.append({
            "name": state["state"],
            "cat": "state",
            "ph": "i",
            "s": "t",
            "ts": state["state_ts"],
            "pid": pid_str,
            "tid": tid,
            "args": {
                "state_id": state["state_id"],
                "proxy_id": proxy_id,
                "coll_sn": coll_sn,
            },
        })

    # Process ProxyStep sub-events: emit their states as instants
    for step_ev in ev.get("events", []):
        if step_ev["event_type"] != "ProxyStep":
            continue

        for state in step_ev.get("proxy_states", []):
            s_args = {
                "state_id": state["state_id"],
                "proxy_id": proxy_id,
                "step": step_ev["proxy_step"],
                "coll_sn": coll_sn,
            }
            if "trans_size" in state:
                s_args["trans_size"] = state["trans_size"]
            trace_events.append({
                "name": state["state"],
                "cat": "state",
                "ph": "i",
                "s": "t",
                "ts": state["state_ts"],
                "pid": pid_str,
                "tid": tid,
                "args": s_args,
            })


def build_metadata_events(records_by_key):
    """Generate process_name, process_sort_index, and thread_name metadata events."""
    meta_events = []

    for (rank, hostname), records in sorted(records_by_key.items()):
        pid_str = make_pid_str(rank, hostname)

        # Process name and sort index
        meta_events.append({
            "name": "process_name",
            "ph": "M",
            "pid": pid_str,
            "tid": "",
            "args": {"name": f"Rank {rank} ({hostname})"},
        })
        meta_events.append({
            "name": "process_sort_index",
            "ph": "M",
            "pid": pid_str,
            "tid": "",
            "args": {"sort_index": rank},
        })

        # Collect all unique tids used by this process
        tids_seen = set()
        for rec in records:
            if "proxy_event" in rec:
                tids_seen.add("ProxyCtrl")
            elif "coll_perf" in rec:
                tids_seen.add("Collective")
                for ev in rec["coll_perf"].get("events", []):
                    if ev["event_type"] == "KernelCh":
                        tids_seen.add("Kernel")
                    elif ev["event_type"] == "ProxyOp":
                        d = ev["proxy_direction"]
                        ch = ev["proxy_channel_id"]
                        p = ev["proxy_peer"]
                        tids_seen.add(f"Proxy ch{ch} {d}\u2192peer{p}")

        # Emit thread_name metadata for each tid
        for tid in sorted(tids_seen):
            meta_events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": pid_str,
                "tid": tid,
                "args": {"name": tid},
            })

    return meta_events


def main():
    parser = argparse.ArgumentParser(
        description="Convert NCCL Inspector profiler logs to Chrome Trace JSON."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Input log file(s) or director(ies) containing .log files"
    )
    parser.add_argument(
        "-o", "--output", default="trace.json",
        help="Output Chrome Trace JSON file (default: trace.json)"
    )
    args = parser.parse_args()

    # Collect all log files
    log_files = collect_log_files(args.inputs)
    if not log_files:
        print("ERROR: no .log files found in specified paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(log_files)} log file(s)...", file=sys.stderr)

    # Parse all records, grouped by (rank, hostname)
    records_by_key = {}
    all_records_with_pid = []  # (pid_str, record) pairs
    for log_file in log_files:
        for rec in parse_log_file(log_file):
            header = rec.get("header", {})
            metadata = rec.get("metadata", {})
            rank = header.get("rank", 0)
            hostname = metadata.get("hostname", "unknown")
            key = (rank, hostname)
            records_by_key.setdefault(key, []).append(rec)
            pid_str = make_pid_str(rank, hostname)
            all_records_with_pid.append((pid_str, rec))

    if not all_records_with_pid:
        print("ERROR: no valid records found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Total records: {len(all_records_with_pid)}", file=sys.stderr)
    print(f"  Processes: {len(records_by_key)}", file=sys.stderr)

    # Convert records to Chrome Trace events
    trace_events = []
    for pid_str, rec in all_records_with_pid:
        if "proxy_event" in rec:
            convert_proxy_ctrl_event(rec, trace_events, pid_str)
        elif "coll_perf" in rec:
            convert_coll_perf(rec, trace_events, pid_str)

    # Add metadata events at the front
    meta_events = build_metadata_events(records_by_key)
    trace_events = meta_events + trace_events

    # Build output
    output = {
        "traceEvents": trace_events,
        "displayTimeUnit": "us",
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Output: {output_path} ({file_size_mb:.2f} MB, {len(trace_events)} events)",
          file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
