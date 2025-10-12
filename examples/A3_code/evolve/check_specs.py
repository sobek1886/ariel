# check_specs.py
import multiprocessing
import psutil
import time

print("=" * 60)
print("SYSTEM SPECIFICATIONS")
print("=" * 60)

# CPU Info
cpu_count = multiprocessing.cpu_count()
print(f"\n💻 CPU Information:")
print(f"   Total CPU cores: {cpu_count}")
print(f"   Physical cores: {psutil.cpu_count(logical=False)}")
print(f"   Logical cores (with hyperthreading): {psutil.cpu_count(logical=True)}")

# Check CPU usage
cpu_percent = psutil.cpu_percent(interval=1)
print(f"   Current CPU usage: {cpu_percent}%")

# RAM Info
ram = psutil.virtual_memory()
print(f"\n💾 RAM Information:")
print(f"   Total RAM: {ram.total / (1024**3):.2f} GB")
print(f"   Available RAM: {ram.available / (1024**3):.2f} GB")
print(f"   Used RAM: {ram.used / (1024**3):.2f} GB")
print(f"   RAM usage: {ram.percent}%")

# Disk space
disk = psutil.disk_usage('C:\\' if os.name == 'nt' else '/')
print(f"\n💿 Disk Information:")
print(f"   Total disk: {disk.total / (1024**3):.2f} GB")
print(f"   Free disk: {disk.free / (1024**3):.2f} GB")

print("\n" + "=" * 60)
print("EVOLUTION CAPACITY ESTIMATES")
print("=" * 60)

# Calculate what you can run
usable_cores = cpu_count
available_ram_gb = ram.available / (1024**3)

# Conservative RAM estimate: 200MB per robot
max_pop_by_ram = int(available_ram_gb * 1000 / 200)

print(f"\n📊 Recommendations based on your hardware:")
print(f"   Cores available: {usable_cores}")
print(f"   Max population (RAM limited): {max_pop_by_ram}")

# Time estimates
print(f"\n⏱️ Overnight (8 hours = 28,800 sec) capacity:")

# Assuming ~5 sec per generation with 20 pop
# Scales linearly: (pop_size / 20) * 5 sec
scenarios = [
    (50, 500),
    (100, 300),
    (100, 500),
    (150, 200),
    (200, 150),
]

for pop, gens in scenarios:
    time_per_gen = (pop / 20) * 5  # Scale from your baseline
    total_time = time_per_gen * gens
    total_evals = pop * gens
    hours = total_time / 3600
    
    if total_time < 28800:  # Fits in 8 hours
        status = "✅ FITS"
    else:
        status = "❌ TOO LONG"
    
    print(f"   {pop:3d} pop × {gens:3d} gens = {total_time/3600:.1f}h ({total_evals:,} evals) {status}")

print("\n" + "=" * 60)