import json
import statistics
import collections
import networkx as nx


WORKFLOW_FEATURE_NAMES = [
    "ACTUAL_NUM_TASKS", "NUM_EDGES",
    "DAG_DEPTH", "MAX_PARALLEL_WIDTH", "CRITICAL_PATH_TASKS",
    "NUM_TASK_CATEGORIES", "TOTAL_INPUT_BYTES",
]

PLATFORM_FEATURE_NAMES = [
    "PLATFORM_NUM_HOSTS", "PLATFORM_CORES_PER_HOST", "PLATFORM_TOTAL_CORES",
    "PLATFORM_HOST_SPEED_GF", "PLATFORM_NET_BW_MBPS", "PLATFORM_NET_LATENCY_MS",
    "PLATFORM_DISK_BW_MBPS", "PLATFORM_WATT_IDLE", "PLATFORM_WATT_BUSY",
]

PLATFORM_CONFIGS = [
    {"name": "lenovo_st45", "PLATFORM_NUM_HOSTS": 1, "PLATFORM_HOST_SPEED_GF": 40.8,
     "PLATFORM_CORES_PER_HOST": 12, "PLATFORM_TOTAL_CORES": 12,
     "PLATFORM_NET_BW_MBPS": 3125, "PLATFORM_NET_LATENCY_MS": 0.01,
     "PLATFORM_DISK_BW_MBPS": 1000, "PLATFORM_WATT_IDLE": 21.5, "PLATFORM_WATT_BUSY": 105},
    {"name": "lenovo_thinkedge", "PLATFORM_NUM_HOSTS": 2, "PLATFORM_HOST_SPEED_GF": 32.0,
     "PLATFORM_CORES_PER_HOST": 16, "PLATFORM_TOTAL_CORES": 32,
     "PLATFORM_NET_BW_MBPS": 3125, "PLATFORM_NET_LATENCY_MS": 0.01,
     "PLATFORM_DISK_BW_MBPS": 1000, "PLATFORM_WATT_IDLE": 44.7, "PLATFORM_WATT_BUSY": 89},
    {"name": "dell_r6715", "PLATFORM_NUM_HOSTS": 1, "PLATFORM_HOST_SPEED_GF": 336.0,
     "PLATFORM_CORES_PER_HOST": 160, "PLATFORM_TOTAL_CORES": 160,
     "PLATFORM_NET_BW_MBPS": 3125, "PLATFORM_NET_LATENCY_MS": 0.01,
     "PLATFORM_DISK_BW_MBPS": 1000, "PLATFORM_WATT_IDLE": 72.8, "PLATFORM_WATT_BUSY": 374},
    {"name": "dell_r6725", "PLATFORM_NUM_HOSTS": 1, "PLATFORM_HOST_SPEED_GF": 672.0,
     "PLATFORM_CORES_PER_HOST": 320, "PLATFORM_TOTAL_CORES": 320,
     "PLATFORM_NET_BW_MBPS": 3125, "PLATFORM_NET_LATENCY_MS": 0.01,
     "PLATFORM_DISK_BW_MBPS": 1000, "PLATFORM_WATT_IDLE": 135, "PLATFORM_WATT_BUSY": 711},
    {"name": "lenovo_sr850", "PLATFORM_NUM_HOSTS": 1, "PLATFORM_HOST_SPEED_GF": 688.0,
     "PLATFORM_CORES_PER_HOST": 344, "PLATFORM_TOTAL_CORES": 344,
     "PLATFORM_NET_BW_MBPS": 3125, "PLATFORM_NET_LATENCY_MS": 0.01,
     "PLATFORM_DISK_BW_MBPS": 1000, "PLATFORM_WATT_IDLE": 450, "PLATFORM_WATT_BUSY": 1578},
]


def extract_workflow_features(wf_json: dict) -> dict:
    spec = wf_json.get("workflow", {}).get("specification", {})
    exec_data = wf_json.get("workflow", {}).get("execution", {})
    exec_tasks = exec_data.get("tasks", [])
    spec_tasks = spec.get("tasks", [])
    spec_files = spec.get("files", [])

    actual_num_tasks = len(spec_tasks)
    parents_map = {}
    children_map = {}
    for t in spec_tasks:
        tid = t.get("id", "")
        parents_map[tid] = t.get("parents", [])
        children_map[tid] = t.get("children", [])

    num_edges = sum(len(v) for v in children_map.values())

    file_sizes = [f.get("sizeInBytes", 0) or 0 for f in spec_files]
    total_input_bytes = sum(file_sizes)

    categories = set()
    for t in spec_tasks:
        cat = t.get("name") or t.get("id", "unknown")
        categories.add(cat.split("_")[0] if "_" in cat else cat)

    G = nx.DiGraph()
    for t in spec_tasks:
        G.add_node(t["id"])
    for t in spec_tasks:
        for child in t.get("children", []):
            G.add_edge(t["id"], child)

    dag_depth = 0
    max_parallel_width = 0
    critical_path_tasks = 0

    if G.number_of_nodes() > 0:
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            topo_order = list(G.nodes())

        dist_tasks = {}
        for node in topo_order:
            preds = list(G.predecessors(node))
            if not preds:
                dist_tasks[node] = 1
            else:
                best_pred = max(preds, key=lambda p: dist_tasks.get(p, 0))
                dist_tasks[node] = dist_tasks.get(best_pred, 0) + 1
        if dist_tasks:
            critical_path_tasks = max(dist_tasks.values())

        try:
            dag_depth = nx.dag_longest_path_length(G)
        except Exception:
            dag_depth = 0

        level = {}
        for node in topo_order:
            preds = list(G.predecessors(node))
            level[node] = 0 if not preds else max(level.get(p, 0) for p in preds) + 1
        if level:
            level_counts = collections.Counter(level.values())
            max_parallel_width = max(level_counts.values())

    return {
        "ACTUAL_NUM_TASKS": actual_num_tasks,
        "NUM_EDGES": num_edges,
        "DAG_DEPTH": dag_depth,
        "MAX_PARALLEL_WIDTH": max_parallel_width,
        "CRITICAL_PATH_TASKS": critical_path_tasks,
        "NUM_TASK_CATEGORIES": len(categories),
        "TOTAL_INPUT_BYTES": total_input_bytes,
    }
