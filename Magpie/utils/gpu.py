"""
GPU detection and hardware control utilities.
"""

from __future__ import annotations

import subprocess
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU vendor type."""
    AMD = auto()
    NVIDIA = auto()
    UNKNOWN = auto()


@dataclass
class GPUHardwareInfo:
    """GPU hardware information."""
    vendor: GPUVendor
    architecture: Optional[str] = None
    device_id: int = 0
    device_name: Optional[str] = None
    
    # Power information
    power_current_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    power_max_watts: Optional[float] = None
    
    # Frequency information (MHz)
    gpu_clock_current: Optional[int] = None
    gpu_clock_max: Optional[int] = None
    mem_clock_current: Optional[int] = None
    mem_clock_max: Optional[int] = None
    
    # Temperature (Celsius)
    temperature: Optional[float] = None
    
    # Memory (GB)
    memory_total_gb: Optional[float] = None
    memory_used_gb: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vendor": self.vendor.name,
            "architecture": self.architecture,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "power_current_watts": self.power_current_watts,
            "power_limit_watts": self.power_limit_watts,
            "power_max_watts": self.power_max_watts,
            "gpu_clock_current": self.gpu_clock_current,
            "gpu_clock_max": self.gpu_clock_max,
            "mem_clock_current": self.mem_clock_current,
            "mem_clock_max": self.mem_clock_max,
            "temperature": self.temperature,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
        }


@dataclass
class GPUConfig:
    """GPU configuration for power and frequency control."""
    device_id: int = 0
    # Power limit in watts (None = don't change)
    power_limit_watts: Optional[int] = None
    # GPU clock frequency in MHz (min, max), None = don't change
    gpu_clock_mhz: Optional[Tuple[int, int]] = None
    # Memory clock frequency in MHz (min, max), None = don't change
    mem_clock_mhz: Optional[Tuple[int, int]] = None
    # Lock clocks at specific levels (AMD specific: 0-7)
    gpu_clock_level: Optional[int] = None
    mem_clock_level: Optional[int] = None


class GPUController:
    """GPU hardware controller for AMD and NVIDIA GPUs."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.vendor, self.arch = detect_gpu()
    
    def get_hardware_info(self) -> GPUHardwareInfo:
        """Get detailed hardware information."""
        if self.vendor == GPUVendor.AMD:
            return self._get_amd_info()
        elif self.vendor == GPUVendor.NVIDIA:
            return self._get_nvidia_info()
        return GPUHardwareInfo(vendor=GPUVendor.UNKNOWN)
    
    def _get_amd_info(self) -> GPUHardwareInfo:
        """Get AMD GPU info using rocm-smi."""
        info = GPUHardwareInfo(
            vendor=GPUVendor.AMD,
            architecture=self.arch,
            device_id=self.device_id
        )
        
        try:
            # Get power info
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showpower"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Parse: "GPU[0]		: Average Graphics Package Power (W): 150.0"
                for line in result.stdout.split('\n'):
                    if 'Power' in line and 'W' in line:
                        match = re.search(r'(\d+\.?\d*)\s*$', line)
                        if match:
                            info.power_current_watts = float(match.group(1))
                            break
            
            # Get clock info
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showclocks"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'sclk' in line.lower():
                        match = re.search(r'(\d+)', line)
                        if match:
                            info.gpu_clock_current = int(match.group(1))
                    elif 'mclk' in line.lower():
                        match = re.search(r'(\d+)', line)
                        if match:
                            info.mem_clock_current = int(match.group(1))
            
            # Get temperature
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showtemp"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Temperature' in line:
                        match = re.search(r'(\d+\.?\d*)', line)
                        if match:
                            info.temperature = float(match.group(1))
                            break
            
            # Get memory info
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Total' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            info.memory_total_gb = int(match.group(1)) / (1024**3)
                    elif 'Used' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            info.memory_used_gb = int(match.group(1)) / (1024**3)
                            
        except Exception as e:
            logger.warning(f"Error getting AMD GPU info: {e}")
        
        return info
    
    def _get_nvidia_info(self) -> GPUHardwareInfo:
        """Get NVIDIA GPU info using nvidia-smi."""
        info = GPUHardwareInfo(
            vendor=GPUVendor.NVIDIA,
            architecture=self.arch,
            device_id=self.device_id
        )
        
        try:
            # Query multiple properties at once
            result = subprocess.run(
                [
                    "nvidia-smi", "-i", str(self.device_id),
                    "--query-gpu=name,power.draw,power.limit,power.max_limit,"
                    "clocks.current.graphics,clocks.max.graphics,"
                    "clocks.current.memory,clocks.max.memory,"
                    "temperature.gpu,memory.total,memory.used",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                values = [v.strip() for v in result.stdout.strip().split(',')]
                if len(values) >= 11:
                    info.device_name = values[0]
                    info.power_current_watts = float(values[1]) if values[1] != '[N/A]' else None
                    info.power_limit_watts = float(values[2]) if values[2] != '[N/A]' else None
                    info.power_max_watts = float(values[3]) if values[3] != '[N/A]' else None
                    info.gpu_clock_current = int(values[4]) if values[4] != '[N/A]' else None
                    info.gpu_clock_max = int(values[5]) if values[5] != '[N/A]' else None
                    info.mem_clock_current = int(values[6]) if values[6] != '[N/A]' else None
                    info.mem_clock_max = int(values[7]) if values[7] != '[N/A]' else None
                    info.temperature = float(values[8]) if values[8] != '[N/A]' else None
                    info.memory_total_gb = float(values[9]) / 1024 if values[9] != '[N/A]' else None
                    info.memory_used_gb = float(values[10]) / 1024 if values[10] != '[N/A]' else None
                    
        except Exception as e:
            logger.warning(f"Error getting NVIDIA GPU info: {e}")
        
        return info
    
    def apply_config(self, config: GPUConfig) -> bool:
        """Apply GPU configuration (power limit, frequencies)."""
        if self.vendor == GPUVendor.AMD:
            return self._apply_amd_config(config)
        elif self.vendor == GPUVendor.NVIDIA:
            return self._apply_nvidia_config(config)
        return False
    
    def _apply_amd_config(self, config: GPUConfig) -> bool:
        """Apply configuration to AMD GPU using rocm-smi."""
        success = True
        device = str(config.device_id)
        
        try:
            # Set power limit (overdrive percentage, not direct watts on some GPUs)
            if config.power_limit_watts is not None:
                result = subprocess.run(
                    ["rocm-smi", "-d", device, "--setpoweroverdrive", str(config.power_limit_watts)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set power limit: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set power limit to {config.power_limit_watts}W")
            
            # Set GPU clock level
            if config.gpu_clock_level is not None:
                result = subprocess.run(
                    ["rocm-smi", "-d", device, "--setsclk", str(config.gpu_clock_level)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set GPU clock: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set GPU clock level to {config.gpu_clock_level}")
            
            # Set memory clock level
            if config.mem_clock_level is not None:
                result = subprocess.run(
                    ["rocm-smi", "-d", device, "--setmclk", str(config.mem_clock_level)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set memory clock: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set memory clock level to {config.mem_clock_level}")
                    
        except Exception as e:
            logger.error(f"Error applying AMD config: {e}")
            success = False
        
        return success
    
    def _apply_nvidia_config(self, config: GPUConfig) -> bool:
        """Apply configuration to NVIDIA GPU using nvidia-smi."""
        success = True
        device = str(config.device_id)
        
        try:
            # Set power limit
            if config.power_limit_watts is not None:
                result = subprocess.run(
                    ["nvidia-smi", "-i", device, "-pl", str(config.power_limit_watts)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set power limit: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set power limit to {config.power_limit_watts}W")
            
            # Lock GPU clocks
            if config.gpu_clock_mhz is not None:
                min_clk, max_clk = config.gpu_clock_mhz
                result = subprocess.run(
                    ["nvidia-smi", "-i", device, "-lgc", f"{min_clk},{max_clk}"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    logger.error(f"Failed to lock GPU clocks: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Locked GPU clocks to {min_clk}-{max_clk} MHz")
            
            # Lock memory clocks
            if config.mem_clock_mhz is not None:
                min_clk, max_clk = config.mem_clock_mhz
                result = subprocess.run(
                    ["nvidia-smi", "-i", device, "-lmc", f"{min_clk},{max_clk}"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    logger.error(f"Failed to lock memory clocks: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Locked memory clocks to {min_clk}-{max_clk} MHz")
                    
        except Exception as e:
            logger.error(f"Error applying NVIDIA config: {e}")
            success = False
        
        return success
    
    def reset_config(self) -> bool:
        """Reset GPU to default settings."""
        if self.vendor == GPUVendor.AMD:
            try:
                subprocess.run(
                    ["rocm-smi", "-d", str(self.device_id), "--resetclocks"],
                    capture_output=True, text=True, timeout=10
                )
                logger.info(f"Reset AMD GPU {self.device_id} clocks")
                return True
            except Exception as e:
                logger.error(f"Error resetting AMD GPU: {e}")
                return False
        elif self.vendor == GPUVendor.NVIDIA:
            try:
                # Reset GPU clocks
                subprocess.run(
                    ["nvidia-smi", "-i", str(self.device_id), "-rgc"],
                    capture_output=True, text=True, timeout=10
                )
                # Reset memory clocks
                subprocess.run(
                    ["nvidia-smi", "-i", str(self.device_id), "-rmc"],
                    capture_output=True, text=True, timeout=10
                )
                logger.info(f"Reset NVIDIA GPU {self.device_id} clocks")
                return True
            except Exception as e:
                logger.error(f"Error resetting NVIDIA GPU: {e}")
                return False
        return False


def detect_gpu() -> Tuple[GPUVendor, Optional[str]]:
    """
    Auto-detect GPU vendor and architecture.
    
    Returns:
        Tuple of (GPUVendor, architecture_string)
        - For AMD: ("gfx90a", "gfx942", etc.)
        - For NVIDIA: ("sm_80", "sm_90", etc.)
    """
    # Try AMD GPU first
    amd_arch = _detect_amd_gpu()
    if amd_arch:
        return GPUVendor.AMD, amd_arch
    
    # Try NVIDIA GPU
    nvidia_arch = _detect_nvidia_gpu()
    if nvidia_arch:
        return GPUVendor.NVIDIA, nvidia_arch
    
    logger.warning("No GPU detected")
    return GPUVendor.UNKNOWN, None


def _detect_amd_gpu() -> Optional[str]:
    """Detect AMD GPU architecture using rocminfo."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse rocminfo output for gfx architecture
            for line in result.stdout.split('\n'):
                if 'gfx' in line.lower():
                    # Extract gfx version (e.g., gfx90a, gfx942)
                    match = re.search(r'(gfx\w+)', line, re.IGNORECASE)
                    if match:
                        arch = match.group(1).lower()
                        logger.info(f"Detected AMD GPU: {arch}")
                        return arch
    except FileNotFoundError:
        pass  # rocminfo not available
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        logger.debug(f"AMD GPU detection failed: {e}")
    
    return None


def _detect_nvidia_gpu() -> Optional[str]:
    """Detect NVIDIA GPU architecture using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse compute capability (e.g., "8.0" -> "sm_80")
            compute_cap = result.stdout.strip().split('\n')[0]
            if compute_cap:
                major, minor = compute_cap.split('.')
                arch = f"sm_{major}{minor}"
                logger.info(f"Detected NVIDIA GPU: {arch}")
                return arch
    except FileNotFoundError:
        pass  # nvidia-smi not available
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        logger.debug(f"NVIDIA GPU detection failed: {e}")
    
    return None


def get_gpu_info() -> dict:
    """
    Get detailed GPU information.
    
    Returns:
        Dictionary with GPU details
    """
    vendor, arch = detect_gpu()
    
    info = {
        "vendor": vendor.name,
        "architecture": arch,
        "detected": vendor != GPUVendor.UNKNOWN,
    }
    
    if vendor == GPUVendor.AMD:
        info["profiler"] = "rocprof-compute"
        info["compiler"] = "hipcc"
    elif vendor == GPUVendor.NVIDIA:
        info["profiler"] = "ncu"
        info["compiler"] = "nvcc"
    
    return info


def get_gpu_count() -> int:
    """
    Get the number of available GPUs.
    
    Returns:
        Number of GPUs detected (0 if none)
    """
    # Try NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # nvidia-smi returns one line per GPU
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Try AMD
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Count unique GPU IDs (each GPU has multiple lines in output)
            gpu_ids = set()
            for line in result.stdout.split('\n'):
                match = re.match(r'^\s*GPU\[(\d+)\]', line)
                if match:
                    gpu_ids.add(int(match.group(1)))
            if gpu_ids:
                return len(gpu_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return 0


def list_gpus() -> List[GPUHardwareInfo]:
    """
    List all available GPUs with their hardware information.
    
    Returns:
        List of GPUHardwareInfo for each GPU
    """
    count = get_gpu_count()
    if count == 0:
        return []
    
    gpus = []
    for device_id in range(count):
        controller = GPUController(device_id=device_id)
        info = controller.get_hardware_info()
        gpus.append(info)
    
    return gpus


@dataclass
class MultiGPUConfig:
    """Configuration for multiple GPUs."""
    # Default configuration applied to all GPUs unless overridden
    default_config: Optional[GPUConfig] = None
    # Per-GPU configurations (device_id -> config)
    gpu_configs: Dict[int, GPUConfig] = field(default_factory=dict)
    # List of device IDs to manage (None = all available GPUs)
    device_ids: Optional[List[int]] = None
    # Whether to apply configurations in parallel
    parallel: bool = True
    
    def get_config_for_device(self, device_id: int) -> Optional[GPUConfig]:
        """Get configuration for a specific device."""
        if device_id in self.gpu_configs:
            return self.gpu_configs[device_id]
        if self.default_config:
            # Create a copy with the correct device_id
            config = GPUConfig(
                device_id=device_id,
                power_limit_watts=self.default_config.power_limit_watts,
                gpu_clock_mhz=self.default_config.gpu_clock_mhz,
                mem_clock_mhz=self.default_config.mem_clock_mhz,
                gpu_clock_level=self.default_config.gpu_clock_level,
                mem_clock_level=self.default_config.mem_clock_level,
            )
            return config
        return None


class MultiGPUController:
    """Controller for managing multiple GPUs."""
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU controller.
        
        Args:
            device_ids: List of GPU device IDs to manage. 
                       If None, manages all available GPUs.
        """
        self.gpu_count = get_gpu_count()
        
        if device_ids is None:
            self.device_ids = list(range(self.gpu_count))
        else:
            # Validate device IDs
            self.device_ids = [d for d in device_ids if 0 <= d < self.gpu_count]
        
        # Create controllers for each GPU
        self.controllers: Dict[int, GPUController] = {}
        for device_id in self.device_ids:
            self.controllers[device_id] = GPUController(device_id=device_id)
    
    def get_all_hardware_info(self, parallel: bool = True) -> Dict[int, GPUHardwareInfo]:
        """
        Get hardware information for all managed GPUs.
        
        Args:
            parallel: Whether to query GPUs in parallel
            
        Returns:
            Dictionary mapping device_id to GPUHardwareInfo
        """
        results = {}
        
        if parallel and len(self.device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
                futures = {
                    executor.submit(ctrl.get_hardware_info): device_id
                    for device_id, ctrl in self.controllers.items()
                }
                for future in as_completed(futures):
                    device_id = futures[future]
                    try:
                        results[device_id] = future.result()
                    except Exception as e:
                        logger.error(f"Error getting info for GPU {device_id}: {e}")
                        results[device_id] = GPUHardwareInfo(
                            vendor=GPUVendor.UNKNOWN, device_id=device_id
                        )
        else:
            for device_id, controller in self.controllers.items():
                try:
                    results[device_id] = controller.get_hardware_info()
                except Exception as e:
                    logger.error(f"Error getting info for GPU {device_id}: {e}")
                    results[device_id] = GPUHardwareInfo(
                        vendor=GPUVendor.UNKNOWN, device_id=device_id
                    )
        
        return results
    
    def apply_config(self, config: MultiGPUConfig) -> Dict[int, bool]:
        """
        Apply configuration to multiple GPUs.
        
        Args:
            config: Multi-GPU configuration
            
        Returns:
            Dictionary mapping device_id to success status
        """
        results = {}
        device_ids = config.device_ids if config.device_ids else self.device_ids
        
        def apply_single(device_id: int) -> Tuple[int, bool]:
            if device_id not in self.controllers:
                return device_id, False
            gpu_config = config.get_config_for_device(device_id)
            if gpu_config is None:
                return device_id, True  # No config to apply
            return device_id, self.controllers[device_id].apply_config(gpu_config)
        
        if config.parallel and len(device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
                futures = [executor.submit(apply_single, d) for d in device_ids]
                for future in as_completed(futures):
                    device_id, success = future.result()
                    results[device_id] = success
        else:
            for device_id in device_ids:
                _, success = apply_single(device_id)
                results[device_id] = success
        
        return results
    
    def reset_all(self, parallel: bool = True) -> Dict[int, bool]:
        """
        Reset all managed GPUs to default settings.
        
        Args:
            parallel: Whether to reset GPUs in parallel
            
        Returns:
            Dictionary mapping device_id to success status
        """
        results = {}
        
        def reset_single(device_id: int) -> Tuple[int, bool]:
            return device_id, self.controllers[device_id].reset_config()
        
        if parallel and len(self.device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
                futures = [executor.submit(reset_single, d) for d in self.device_ids]
                for future in as_completed(futures):
                    device_id, success = future.result()
                    results[device_id] = success
        else:
            for device_id in self.device_ids:
                _, success = reset_single(device_id)
                results[device_id] = success
        
        return results
    
    def print_summary(self) -> None:
        """Print a summary of all managed GPUs."""
        infos = self.get_all_hardware_info()
        
        print(f"\n{'='*70}")
        print(f"GPU Summary ({len(infos)} GPUs)")
        print(f"{'='*70}")
        
        for device_id in sorted(infos.keys()):
            info = infos[device_id]
            print(f"\nGPU {device_id}: {info.device_name or 'Unknown'}")
            print(f"  Vendor: {info.vendor.name}")
            print(f"  Architecture: {info.architecture or 'N/A'}")
            print(f"  Power: {info.power_current_watts or 'N/A'}W / "
                  f"{info.power_limit_watts or 'N/A'}W (limit)")
            print(f"  GPU Clock: {info.gpu_clock_current or 'N/A'} MHz")
            print(f"  Mem Clock: {info.mem_clock_current or 'N/A'} MHz")
            print(f"  Temperature: {info.temperature or 'N/A'}°C")
            print(f"  Memory: {info.memory_used_gb or 0:.1f} / "
                  f"{info.memory_total_gb or 0:.1f} GB")
        
        print(f"\n{'='*70}\n")


def load_gpu_config_from_dict(config: Dict) -> Tuple[List[int], Optional[MultiGPUConfig]]:
    """
    Load GPU configuration from a config dictionary (parsed from YAML).
    
    This is the unified way to get GPU settings from config.yaml.
    Both scheduler and hardware controller should use this.
    
    Args:
        config: Dictionary containing 'gpu' section from config.yaml
        
    Returns:
        Tuple of (device_ids, MultiGPUConfig or None if hardware control disabled)
        
    Example:
        config = yaml.safe_load(open('config.yaml'))
        device_ids, gpu_config = load_gpu_config_from_dict(config)
    """
    gpu_section = config.get('gpu', {})
    
    # Get device IDs (unified for scheduler and hardware control)
    device_ids = gpu_section.get('device_ids')
    if device_ids is None:
        # Auto-detect all GPUs
        device_ids = list(range(get_gpu_count()))
    
    # Check if hardware control is enabled
    hardware = gpu_section.get('hardware', {})
    if not hardware.get('enabled', False):
        return device_ids, None
    
    # Parse hardware configuration
    default_section = hardware.get('default', {})
    per_gpu_section = hardware.get('per_gpu', {})
    
    # Build default config
    default_config = None
    if default_section:
        power = default_section.get('power', {})
        freq = default_section.get('frequency', {})
        
        gpu_clock_mhz = freq.get('gpu_clock_mhz')
        if gpu_clock_mhz and isinstance(gpu_clock_mhz, list):
            gpu_clock_mhz = tuple(gpu_clock_mhz)
        
        mem_clock_mhz = freq.get('mem_clock_mhz')
        if mem_clock_mhz and isinstance(mem_clock_mhz, list):
            mem_clock_mhz = tuple(mem_clock_mhz)
        
        default_config = GPUConfig(
            power_limit_watts=power.get('limit_watts'),
            gpu_clock_mhz=gpu_clock_mhz,
            mem_clock_mhz=mem_clock_mhz,
            gpu_clock_level=freq.get('gpu_clock_level'),
            mem_clock_level=freq.get('mem_clock_level'),
        )
    
    # Build per-GPU configs
    gpu_configs = {}
    for device_id_str, gpu_cfg in per_gpu_section.items():
        device_id = int(device_id_str)
        power = gpu_cfg.get('power', {})
        freq = gpu_cfg.get('frequency', {})
        
        gpu_clock_mhz = freq.get('gpu_clock_mhz')
        if gpu_clock_mhz and isinstance(gpu_clock_mhz, list):
            gpu_clock_mhz = tuple(gpu_clock_mhz)
        
        mem_clock_mhz = freq.get('mem_clock_mhz')
        if mem_clock_mhz and isinstance(mem_clock_mhz, list):
            mem_clock_mhz = tuple(mem_clock_mhz)
        
        gpu_configs[device_id] = GPUConfig(
            device_id=device_id,
            power_limit_watts=power.get('limit_watts'),
            gpu_clock_mhz=gpu_clock_mhz,
            mem_clock_mhz=mem_clock_mhz,
            gpu_clock_level=freq.get('gpu_clock_level'),
            mem_clock_level=freq.get('mem_clock_level'),
        )
    
    multi_gpu_config = MultiGPUConfig(
        default_config=default_config,
        gpu_configs=gpu_configs,
        device_ids=device_ids,
        parallel=gpu_section.get('parallel', True),
    )
    
    return device_ids, multi_gpu_config


def get_reset_after_benchmark(config: Dict) -> bool:
    """
    Check if GPUs should be reset after benchmark.
    
    Args:
        config: Dictionary containing 'gpu' section from config.yaml
        
    Returns:
        True if reset is enabled
    """
    gpu_section = config.get('gpu', {})
    hardware = gpu_section.get('hardware', {})
    return hardware.get('reset_after_benchmark', True)

