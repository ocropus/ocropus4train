import psutil
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta

# Number of top processes to display
TOP_PROCESSES = 10

# Initialize dictionary for storing process data by PID
process_data = {}

# Create figure and axis for plotting
fig, ax = plt.subplots()

# Set plot properties
ax.set_xlabel('Time (HH:MM:SS)')
ax.set_ylabel('Memory Usage (MB)')
ax.set_title('Memory Usage of Top 10 Processes')

# Function to update the plot
def update_plot():
    ax.clear()

    # Get current time and memory usage of all processes
    current_time = time.time()
    current_datetime = datetime.now()
    process_list = [(p.pid, p.info) for p in psutil.process_iter(['pid', 'name', 'cmdline'])]

    # Update process data for each process
    for pid, info in process_list:
        try:
            process = psutil.Process(pid)
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / 1024 / 1024  # Convert to MB

            if pid in process_data:
                # Process already exists, update memory usage
                process_data[pid]['data'].append((current_datetime, memory_usage))
            else:
                # Process is new, initialize memory usage data
                process_data[pid] = {'cmdline': info['cmdline'], 'data': [(current_datetime, memory_usage)]}

        except psutil.NoSuchProcess:
            # Process no longer exists, remove it from the data
            if pid in process_data:
                del process_data[pid]

    # Remove data older than two hours
    threshold = current_datetime - timedelta(hours=2)
    for pid, process_info in process_data.items():
        data = process_info['data']
        process_info['data'] = [(t, m) for t, m in data if t >= threshold]

    # Sort processes by current memory usage in descending order
    sorted_processes = sorted(process_data.items(), key=lambda x: x[1]['data'][-1][1], reverse=True)

    # Get top processes
    top_processes = sorted_processes[:TOP_PROCESSES]

    # Update process data for the top processes and plot them
    for i, (pid, process_info) in enumerate(top_processes):
        data = process_info['data']

        # Create legend label
        command = os.path.basename(process_info['cmdline'][0])

        # Plot process data
        ax.plot(*zip(*data), label=f'{pid}: {command}')

    # Configure the legend
    ax.legend(loc='lower left')

    # Update the plot
    plt.draw()
    plt.pause(0.001)

# Run the script indefinitely
while True:
    update_plot()
    time.sleep(1)

# Keep the plot open until it is manually closed
plt.show()
