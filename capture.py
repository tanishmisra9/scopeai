"""
ScopeAI capture backend for Analog Discovery 2 (AD2) via WaveForms SDK.
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import byref, c_bool, c_byte, c_char, c_double, c_int
from typing import Any

import numpy as np

SAMPLE_RATE = 10_000
DEFAULT_DURATION_SEC = 6.0
MAX_SUPPLY_VOLTAGE = 5.0
MIN_SUPPLY_VOLTAGE = 0.0

_SIM_BANNER = "SIMULATION MODE - no AD2 detected"

_DWF_STATE_DONE = 2
_ACQMODE_RECORD = 3
_FILTER_DECIMATE = 0

_SIM_FAULTS = {"nominal", "R_too_high", "R_too_low", "cap_missing", "no_oscillation", "chatter"}


class CaptureError(RuntimeError):
    pass


class _ScopeCaptureBackend:

    def __init__(self) -> None:
        self.sample_rate: int = SAMPLE_RATE
        self.simulation_mode: bool = False
        self.connected: bool = False
        self.supply_voltage: float = 5.0
        self.simulated_fault: str = "nominal"
        self.serial: str = "SIMULATED"
        self.device_name: str = "AD2_SIMULATED"
        self._dwf: Any | None = None
        self._hdwf = c_int()

        force_sim = os.getenv("SCOPEAI_SIMULATE", "0") == "1"
        if force_sim:
            self._enable_simulation(reason="SCOPEAI_SIMULATE=1")
            return

        try:
            self._init_hardware()
        except Exception as exc:
            self._enable_simulation(reason=str(exc))

    def _enable_simulation(self, reason: str) -> None:
        self.simulation_mode = True
        self.connected = False
        self.serial = "SIMULATED"
        self.device_name = "AD2_SIMULATED"
        print(f"{_SIM_BANNER} ({reason})")

    def _load_dwf_library(self) -> ctypes.CDLL:
        if os.name == "nt":
            candidates = ["dwf.dll"]
        elif os.uname().sysname.lower() == "darwin":
            candidates = [
                "/Library/Frameworks/dwf.framework/dwf",
                "/usr/local/lib/libdwf.dylib",
                "libdwf.dylib",
            ]
        else:
            candidates = ["libdwf.so", "/usr/lib/libdwf.so", "/usr/local/lib/libdwf.so"]

        last_error: Exception | None = None
        for path in candidates:
            try:
                return ctypes.cdll.LoadLibrary(path)
            except Exception as exc:
                last_error = exc
        raise CaptureError("Could not load WaveForms SDK (libdwf).") from last_error

    def _init_hardware(self) -> None:
        self._dwf = self._load_dwf_library()
        self._dwf.FDwfDeviceOpen(c_int(-1), byref(self._hdwf))
        if self._hdwf.value == 0:
            raise CaptureError("No Analog Discovery 2 found.")

        self.connected = True
        self.simulation_mode = False
        self.serial = "AD2"
        self.device_name = "Analog Discovery 2"

        self._configure_power_supply(5.0)
        self._configure_analog_in(duration_sec=DEFAULT_DURATION_SEC)

    def _configure_power_supply(self, voltage: float) -> None:
        if self._dwf is None:
            return
        v = float(np.clip(voltage, MIN_SUPPLY_VOLTAGE, MAX_SUPPLY_VOLTAGE))
        # enable V+ channel, then set voltage
        self._dwf.FDwfAnalogIOChannelNodeSet(self._hdwf, c_int(0), c_int(0), c_double(1.0))
        self._dwf.FDwfAnalogIOChannelNodeSet(self._hdwf, c_int(0), c_int(1), c_double(v))
        self._dwf.FDwfAnalogIOEnableSet(self._hdwf, c_int(1))
        self._dwf.FDwfAnalogIOConfigure(self._hdwf)
        self.supply_voltage = v
        time.sleep(0.5)

    def _configure_analog_in(self, duration_sec: float) -> None:
        if self._dwf is None:
            return
        n_samples = max(16, int(round(duration_sec * self.sample_rate)))
        self._dwf.FDwfAnalogInReset(self._hdwf)
        self._dwf.FDwfAnalogInChannelEnableSet(self._hdwf, c_int(0), c_bool(True))
        self._dwf.FDwfAnalogInChannelRangeSet(self._hdwf, c_int(0), c_double(5.0))
        self._dwf.FDwfAnalogInChannelOffsetSet(self._hdwf, c_int(0), c_double(0.0))
        self._dwf.FDwfAnalogInChannelFilterSet(self._hdwf, c_int(0), c_int(_FILTER_DECIMATE))
        self._dwf.FDwfAnalogInAcquisitionModeSet(self._hdwf, c_int(_ACQMODE_RECORD))
        self._dwf.FDwfAnalogInFrequencySet(self._hdwf, c_double(float(self.sample_rate)))
        self._dwf.FDwfAnalogInRecordLengthSet(self._hdwf, c_double(float(duration_sec)))
        self._dwf.FDwfAnalogInBufferSizeSet(self._hdwf, c_int(n_samples))

    def _capture_hardware(self, duration_sec: float) -> np.ndarray:
        if self._dwf is None:
            raise CaptureError("DWF library not initialized.")

        duration_sec = max(0.05, float(duration_sec))
        n_samples = max(1, int(round(duration_sec * self.sample_rate)))
        self._configure_analog_in(duration_sec=duration_sec)

        samples = (c_double * n_samples)()
        total_read = 0
        status = c_byte()

        self._dwf.FDwfAnalogInConfigure(self._hdwf, c_int(0), c_int(1))
        start = time.time()
        timeout_sec = duration_sec + 2.0

        while total_read < n_samples and (time.time() - start) < timeout_sec:
            self._dwf.FDwfAnalogInStatus(self._hdwf, c_int(1), byref(status))
            available = c_int()
            lost = c_int()
            corrupted = c_int()
            self._dwf.FDwfAnalogInStatusRecord(
                self._hdwf, byref(available), byref(lost), byref(corrupted)
            )
            if available.value > 0:
                chunk = min(available.value, n_samples - total_read)
                byte_offset = total_read * ctypes.sizeof(c_double)
                self._dwf.FDwfAnalogInStatusData(
                    self._hdwf, c_int(0), byref(samples, byte_offset), c_int(chunk)
                )
                total_read += chunk
            if status.value == _DWF_STATE_DONE and available.value <= 0:
                break
            time.sleep(0.003)

        if total_read <= 0:
            raise CaptureError("AD2 capture returned no samples.")

        arr = np.ctypeslib.as_array(samples)[:total_read].astype(np.float64, copy=False)
        if arr.size < n_samples:
            pad_val = float(arr[-1]) if arr.size else 0.0
            arr = np.pad(arr, (0, n_samples - arr.size), mode="constant", constant_values=pad_val)
        return arr

    def _smooth_noise(self, n_samples: int, kernel_size: int) -> np.ndarray:
        kernel = np.ones(max(3, kernel_size), dtype=np.float64)
        kernel /= float(kernel.size)
        white = np.random.normal(0.0, 1.0, size=n_samples)
        return np.convolve(white, kernel, mode="same")

    def _simulate_waveform(self, duration_sec: float) -> np.ndarray:
        n_samples = max(1, int(round(max(0.05, duration_sec) * self.sample_rate)))
        t = np.arange(n_samples, dtype=np.float64) / float(self.sample_rate)

        if self.simulated_fault == "nominal":
            freq = 2.5
            wave = np.where(np.mod(t * freq, 1.0) < 0.6, 1.0, 0.0)
            volts = 0.1 + 1.8 * wave + np.random.normal(0.0, 0.01, size=n_samples)
        elif self.simulated_fault == "R_too_high":
            freq = 0.5
            wave = np.where(np.mod(t * freq, 1.0) < 0.55, 1.0, 0.0)
            volts = 0.1 + 1.8 * wave + np.random.normal(0.0, 0.01, size=n_samples)
        elif self.simulated_fault == "R_too_low":
            freq = 7.0
            wave = np.where(np.mod(t * freq, 1.0) < 0.95, 1.0, 0.0)
            volts = 0.1 + 1.8 * wave + np.random.normal(0.0, 0.01, size=n_samples)
        elif self.simulated_fault == "cap_missing":
            volts = np.full(n_samples, 0.08) + np.random.normal(0.0, 0.001, size=n_samples)
        elif self.simulated_fault == "no_oscillation":
            volts = np.full(n_samples, 0.01) + np.random.normal(0.0, 0.001, size=n_samples)
        else:
            volts = np.random.normal(0.5, 0.1, size=n_samples)

        return np.clip(volts, 0.0, 5.0).astype(np.float64)

    def capture(self, duration_sec: float = DEFAULT_DURATION_SEC) -> tuple[np.ndarray, int]:
        if self.simulation_mode:
            return self._simulate_waveform(duration_sec), self.sample_rate
        return self._capture_hardware(duration_sec), self.sample_rate

    def capture_snapshot(self, duration_sec: float = 6.0) -> tuple[np.ndarray, int]:
        return self.capture(duration_sec=duration_sec)

    def set_supply_voltage(self, voltage: float) -> str:
        clamped = float(np.clip(voltage, MIN_SUPPLY_VOLTAGE, MAX_SUPPLY_VOLTAGE))
        self.supply_voltage = clamped
        if not self.simulation_mode:
            self._configure_power_supply(clamped)
        return f"Supply voltage set to {clamped:.2f}V."

    def get_supply_voltage(self) -> float:
        return float(self.supply_voltage)

    def set_simulated_fault(self, fault_class: str) -> str:
        if fault_class not in _SIM_FAULTS:
            raise ValueError(f'Invalid fault "{fault_class}".')
        self.simulated_fault = fault_class
        return f"Simulated fault set to {fault_class}."

    def get_device_info(self) -> dict[str, Any]:
        return {
            "connected": bool(self.connected),
            "simulation_mode": bool(self.simulation_mode),
            "device_name": self.device_name,
            "serial": self.serial,
            "sample_rate": self.sample_rate,
            "supply_voltage": float(self.supply_voltage),
            "simulated_fault": self.simulated_fault if self.simulation_mode else None,
        }

    def close(self) -> None:
        if self.simulation_mode or self._dwf is None:
            return
        try:
            self._dwf.FDwfAnalogIOEnableSet(self._hdwf, c_int(0))
            self._dwf.FDwfAnalogIOConfigure(self._hdwf)
        except Exception:
            pass
        try:
            self._dwf.FDwfDeviceClose(self._hdwf)
        except Exception:
            pass
        self.connected = False


_BACKEND = _ScopeCaptureBackend()


def capture(duration_sec: float = DEFAULT_DURATION_SEC) -> tuple[np.ndarray, int]:
    return _BACKEND.capture(duration_sec=duration_sec)


def capture_snapshot(duration_sec: float = 6.0) -> tuple[np.ndarray, int]:
    return _BACKEND.capture_snapshot(duration_sec=duration_sec)


def set_supply_voltage(voltage: float) -> str:
    return _BACKEND.set_supply_voltage(voltage=voltage)


def get_supply_voltage() -> float:
    return _BACKEND.get_supply_voltage()


def set_simulated_fault(fault_class: str) -> str:
    return _BACKEND.set_simulated_fault(fault_class=fault_class)


def get_device_info() -> dict[str, Any]:
    return _BACKEND.get_device_info()


def close() -> None:
    _BACKEND.close()
