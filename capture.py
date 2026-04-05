"""
ScopeAI capture backend for Analog Discovery 2 (AD2) via WaveForms SDK.

This module exposes a simple API for waveform capture and power-supply control:
- capture()
- capture_snapshot()
- set_supply_voltage()
- get_supply_voltage()
- get_device_info()
- set_simulated_fault()
- close()

If AD2 is not available, or SCOPEAI_SIMULATE=1 is set, it runs in simulation mode.
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import byref, c_bool, c_byte, c_char, c_double, c_int
from typing import Any

import numpy as np

SAMPLE_RATE = 44100
DEFAULT_DURATION_SEC = 1.0
MAX_SUPPLY_VOLTAGE = 5.0
MIN_SUPPLY_VOLTAGE = 0.0

_SIM_BANNER = "SIMULATION MODE - no AD2 detected"

# DWF constants (from WaveForms SDK)
_DWF_STATE_DONE = 2
_ACQMODE_RECORD = 3
_FILTER_DECIMATE = 0

_SIM_FAULTS = {"nominal", "R_too_high", "R_too_low", "cap_missing", "no_oscillation", "chatter"}


class CaptureError(RuntimeError):
    """Raised when capture backend encounters an unrecoverable issue."""


class _ScopeCaptureBackend:
    """Stateful AD2/simulation backend for ScopeAI waveform capture."""

    def __init__(self) -> None:
        """Initialize backend and choose hardware or simulation mode."""
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
        """Switch backend to simulation mode and print a startup banner."""
        self.simulation_mode = True
        self.connected = False
        self.serial = "SIMULATED"
        self.device_name = "AD2_SIMULATED"
        print(f"{_SIM_BANNER} ({reason})")

    def _load_dwf_library(self) -> ctypes.CDLL:
        """Load WaveForms DWF shared library for the current platform."""
        if os.name == "nt":
            candidates = ["dwf.dll"]
        elif os.uname().sysname.lower() == "darwin":
            candidates = [
                "/Library/Frameworks/dwf.framework/dwf",
                "/usr/local/lib/libdwf.dylib",
                "/opt/homebrew/lib/libdwf.dylib",
                "libdwf.dylib",
            ]
        else:
            candidates = ["libdwf.so", "/usr/lib/libdwf.so", "/usr/local/lib/libdwf.so"]

        last_error: Exception | None = None
        for path in candidates:
            try:
                return ctypes.cdll.LoadLibrary(path)
            except Exception as exc:  # pragma: no cover - hardware-dependent
                last_error = exc
        raise CaptureError(
            "Could not load WaveForms SDK (libdwf). Install Digilent WaveForms SDK."
        ) from last_error

    def _init_hardware(self) -> None:
        """Initialize AD2 hardware connection and configure analog I/O."""
        self._dwf = self._load_dwf_library()

        # Open first device. -1 means first available.
        self._dwf.FDwfDeviceOpen(c_int(-1), byref(self._hdwf))
        if self._hdwf.value == 0:
            raise CaptureError(
                "No Analog Discovery 2 found. Connect AD2 via USB or set SCOPEAI_SIMULATE=1."
            )

        self.connected = True
        self.simulation_mode = False
        self.serial = self._get_serial_or_unknown()
        self.device_name = "Analog Discovery 2"

        self._configure_power_supply(5.0)
        self._configure_analog_in(duration_sec=DEFAULT_DURATION_SEC)

    def _get_serial_or_unknown(self) -> str:
        """Read AD2 serial string if available; otherwise return placeholder."""
        if self._dwf is None:
            return "UNKNOWN"
        serial_buf = (c_char * 32)()
        # Device index 0 = first enumerated.
        ok = self._dwf.FDwfEnumSN(c_int(0), serial_buf)
        if ok:
            serial = bytes(serial_buf).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            return serial or "UNKNOWN"
        return "UNKNOWN"

    def _configure_power_supply(self, voltage: float) -> None:
        """Set AD2 positive supply voltage and enable power output."""
        if self._dwf is None:
            return
        v = float(np.clip(voltage, MIN_SUPPLY_VOLTAGE, MAX_SUPPLY_VOLTAGE))
        self._dwf.FDwfAnalogIOChannelNodeSet(self._hdwf, c_int(0), c_int(0), c_double(v))
        self._dwf.FDwfAnalogIOChannelNodeSet(self._hdwf, c_int(0), c_int(1), c_double(1.0))
        self._dwf.FDwfAnalogIOEnableSet(self._hdwf, c_int(1))
        self._dwf.FDwfAnalogIOConfigure(self._hdwf)
        self.supply_voltage = v

    def _configure_analog_in(self, duration_sec: float) -> None:
        """Configure AD2 analog input channel for record-mode capture."""
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
        """Capture waveform samples from AD2 in record mode."""
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
            raise CaptureError(
                "AD2 capture returned no samples. Check probe wiring (CH1+/CH1-) and signal."
            )

        arr = np.ctypeslib.as_array(samples)[:total_read].astype(np.float64, copy=False)
        if arr.size < n_samples:
            # Pad with last value for fixed-length downstream assumptions.
            pad_val = float(arr[-1]) if arr.size else 0.0
            arr = np.pad(arr, (0, n_samples - arr.size), mode="constant", constant_values=pad_val)
        return arr

    def _smooth_noise(self, n_samples: int, kernel_size: int) -> np.ndarray:
        """Return low-pass-like noise by convolving white noise with a box kernel."""
        kernel = np.ones(max(3, kernel_size), dtype=np.float64)
        kernel /= float(kernel.size)
        white = np.random.normal(0.0, 1.0, size=n_samples)
        return np.convolve(white, kernel, mode="same")

    def _simulate_waveform(self, duration_sec: float) -> np.ndarray:
        """Generate synthetic waveform data for simulation mode."""
        n_samples = max(1, int(round(max(0.05, duration_sec) * self.sample_rate)))
        t = np.arange(n_samples, dtype=np.float64) / float(self.sample_rate)

        # Keep within 0-5V style domain for NE555 demo realism.
        v_mid = self.supply_voltage / 2.0
        amplitude = min(2.0, max(0.02, 0.4 * self.supply_voltage))
        noise_std = 0.03

        if self.simulated_fault == "nominal":
            freq = 28.0
            duty = 0.52
            wave = np.where(np.mod(t * freq, 1.0) < duty, 1.0, -1.0)
        elif self.simulated_fault == "R_too_high":
            freq = 2.5
            duty = 0.62
            wave = np.where(np.mod(t * freq, 1.0) < duty, 1.0, -1.0)
            noise_std = 0.02
        elif self.simulated_fault == "R_too_low":
            freq = 58.0
            duty = 0.44
            wave = np.where(np.mod(t * freq, 1.0) < duty, 1.0, -1.0)
            noise_std = 0.025
        elif self.simulated_fault == "cap_missing":
            # Cap missing: weak near-DC behavior plus mains pickup and tiny fast chatter.
            # This keeps amplitude tiny while preventing broadband white-noise outliers.
            base_dc = 0.08 * self.supply_voltage
            hum_60hz = 0.0011 * np.sin(2.0 * np.pi * 60.0 * t)
            tiny_chatter = 0.00017 * np.sign(np.sin(2.0 * np.pi * 650.0 * t))
            drift = 0.0006 * self._smooth_noise(n_samples=n_samples, kernel_size=110)
            hiss = 0.00014 * np.random.normal(0.0, 1.0, size=n_samples)
            volts = base_dc + hum_60hz + tiny_chatter + drift + hiss
            volts = np.clip(volts, 0.0, max(0.1, self.supply_voltage))
            return volts.astype(np.float64, copy=False)
        elif self.simulated_fault == "no_oscillation":
            # No oscillation: pinned output with ripple/noise; keep dominant energy near 60 Hz.
            # Add mild FM so timing metrics do not collapse to perfect periodicity.
            base_dc = 0.90
            inst_freq = 60.0 + 10.0 * np.sin(2.0 * np.pi * 2.0 * t) + 3.0 * np.sin(
                2.0 * np.pi * 4.1 * t
            )
            phase_60 = 2.0 * np.pi * np.cumsum(inst_freq) / float(self.sample_rate)
            hum_60hz = 0.82 * np.sin(phase_60)
            harmonic_180 = 0.12 * np.sin(2.0 * np.pi * 180.0 * t + 0.4)
            flicker = 0.025 * self._smooth_noise(n_samples=n_samples, kernel_size=128)
            hiss = 0.020 * np.random.normal(0.0, 1.0, size=n_samples)
            volts = base_dc + hum_60hz + harmonic_180 + flicker + hiss
            volts = np.clip(volts, 0.0, max(0.1, self.supply_voltage))
            return volts.astype(np.float64, copy=False)
        else:  # chatter
            base = np.where(np.mod(t * 18.0, 1.0) < 0.5, 1.0, -1.0)
            chatter = 0.6 * np.sin(2.0 * np.pi * 220.0 * t)
            wave = np.clip(base + chatter, -1.0, 1.0)
            noise_std = 0.06

        volts = v_mid + amplitude * wave + np.random.normal(0.0, noise_std, size=n_samples)
        volts = np.clip(volts, 0.0, max(0.1, self.supply_voltage))
        return volts.astype(np.float64, copy=False)

    def capture(self, duration_sec: float = DEFAULT_DURATION_SEC) -> tuple[np.ndarray, int]:
        """Capture a waveform and return `(samples, sample_rate)`."""
        if self.simulation_mode:
            return self._simulate_waveform(duration_sec), self.sample_rate
        return self._capture_hardware(duration_sec), self.sample_rate

    def capture_snapshot(self, duration_sec: float = 0.5) -> tuple[np.ndarray, int]:
        """Capture a shorter waveform snapshot for tool-driven chat interactions."""
        return self.capture(duration_sec=duration_sec)

    def set_supply_voltage(self, voltage: float) -> str:
        """Set supply voltage (clamped 0-5V) and return confirmation text."""
        clamped = float(np.clip(voltage, MIN_SUPPLY_VOLTAGE, MAX_SUPPLY_VOLTAGE))
        self.supply_voltage = clamped
        if not self.simulation_mode:
            self._configure_power_supply(clamped)
        if voltage > MAX_SUPPLY_VOLTAGE:
            return f"Requested {voltage:.2f}V, clamped to {clamped:.2f}V (max 5.00V)."
        return f"Supply voltage set to {clamped:.2f}V."

    def get_supply_voltage(self) -> float:
        """Return current configured supply voltage."""
        return float(self.supply_voltage)

    def set_simulated_fault(self, fault_class: str) -> str:
        """Set the active simulated fault class."""
        if fault_class not in _SIM_FAULTS:
            allowed = ", ".join(sorted(_SIM_FAULTS))
            raise ValueError(f'Invalid simulated fault "{fault_class}". Allowed: {allowed}')
        self.simulated_fault = fault_class
        return f"Simulated fault set to {fault_class}."

    def get_device_info(self) -> dict[str, Any]:
        """Return connected device and current capture/power configuration."""
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
        """Disable outputs and close hardware handle."""
        if self.simulation_mode:
            return
        if self._dwf is None:
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
    """Capture waveform samples for the requested duration."""
    return _BACKEND.capture(duration_sec=duration_sec)


def capture_snapshot(duration_sec: float = 0.5) -> tuple[np.ndarray, int]:
    """Capture a quick snapshot for interactive diagnostics."""
    return _BACKEND.capture_snapshot(duration_sec=duration_sec)


def set_supply_voltage(voltage: float) -> str:
    """Set AD2/simulated V+ supply voltage and return confirmation."""
    return _BACKEND.set_supply_voltage(voltage=voltage)


def get_supply_voltage() -> float:
    """Return current AD2/simulated V+ setting."""
    return _BACKEND.get_supply_voltage()


def set_simulated_fault(fault_class: str) -> str:
    """Select which fault to emulate while in simulation mode."""
    return _BACKEND.set_simulated_fault(fault_class=fault_class)


def get_device_info() -> dict[str, Any]:
    """Return AD2/simulation backend status details."""
    return _BACKEND.get_device_info()


def close() -> None:
    """Shut down backend resources."""
    _BACKEND.close()


if __name__ == "__main__":
    samples, sr = capture(1.0)
    info = get_device_info()
    print(f"Device info: {info}")
    print(
        "Captured stats: "
        f"n={samples.size}, sr={sr}, min={samples.min():.4f}V, "
        f"max={samples.max():.4f}V, mean={samples.mean():.4f}V"
    )
