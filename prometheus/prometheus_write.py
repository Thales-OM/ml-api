from prometheus_client import start_http_server, Gauge, CollectorRegistry
import psutil
import time
import asyncio
from config import PORT_PROMETHEUS_SERVER, ENV, APP_NAME

# Create Prometheus metrics
registry = CollectorRegistry()
if 'total_memory_bytes' not in registry._names_to_collectors:
    TOTAL_MEMORY = Gauge('total_memory_bytes', 'Total available memory in bytes', ['app', 'env'], registry=registry)
if 'memory_usage_ratio' not in registry._names_to_collectors:
    MEMORY_USAGE = Gauge('memory_usage_ratio', 'Memory usage as ration of used to total memory', ['app', 'env'], registry=registry)
if  'cpu_usage_percent' not in registry._names_to_collectors:
    CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage by core', ['app', 'env', 'core'], registry=registry)

# Function to get memory usage
def get_memory_usage():
    memory = psutil.virtual_memory()
    total_memory_bytes = memory.total
    used_memory_bytes = memory.used
    memory_usage = used_memory_bytes/total_memory_bytes
    return total_memory_bytes, memory_usage

# Function to get CPU usage
def get_cpu_usage(percpu=True):
    return psutil.cpu_percent(interval=1, percpu=percpu)

async def collect_metrics():
    while True:
        # Get memory usage
        total_memory_bytes, memory_usage = get_memory_usage()
        # MEMORY_USAGE.clear()
        MEMORY_USAGE.labels(app=APP_NAME, env=ENV).set(memory_usage)
        # TOTAL_MEMORY.clear()
        TOTAL_MEMORY.labels(app=APP_NAME, env=ENV).set(total_memory_bytes)

        # Get CPU usage for each core
        cpu_usage_per_core = psutil.cpu_percent(percpu=True)
        # Clear previous metrics
        # CPU_USAGE.clear()
        # Set the metric for each core
        for core, usage in enumerate(cpu_usage_per_core):
            CPU_USAGE.labels(app=APP_NAME, env=ENV, core=core).set(usage)

        # Sleep for a specified interval
        await asyncio.sleep(5)  # Use asyncio.sleep for non-blocking sleep

if __name__ == '__main__':
    # Start up the server to expose the metrics.
    start_http_server(PORT_PROMETHEUS_SERVER)
    collect_metrics()