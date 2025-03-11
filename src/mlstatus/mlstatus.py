#!/usr/bin/env python3

import argparse
import fcntl
import math
import os
import pty
import selectors
import shutil
import signal
import subprocess
import sys
import struct
import termios
import time
from typing import Optional

import ctypes


UPDATE_PERIOD = 0.5


class TimerFD_timespec(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_int64),
        ("tv_nsec", ctypes.c_long),
    ]

    def __str__(self):
        return f"timespec(sec={self.tv_sec}, nsec={self.tv_nsec})"


class TimerFD_itimerspec(ctypes.Structure):
    _fields_ = [
        ("it_interval", TimerFD_timespec),
        ("it_value", TimerFD_timespec),
    ]


class TimerFD:
    def __init__(self, clock_id: int):
        self._lib = ctypes.CDLL("libc.so.6")

        ret = self._lib.timerfd_create(clock_id, os.O_CLOEXEC)
        assert ret >= 0

        self._fd = ret

    def settime(self, value: float, initial: Optional[float] = None):
        if initial is None:
            initial = value

        spec = TimerFD_itimerspec()
        spec.it_interval.tv_sec = int(value)
        spec.it_interval.tv_nsec = int((value * 1000000000) % 1000000000)
        spec.it_value.tv_sec = int(initial)
        spec.it_value.tv_nsec = int((initial * 1000000000) % 1000000000)

        ret = self._lib.timerfd_settime(self._fd, 0, ctypes.byref(spec), None)
        assert ret == 0

    def fileno(self):
        return self._fd

    def read(self):
        expirations = os.read(self._fd, 8)
        return ctypes.c_uint64.from_buffer_copy(expirations).value


################################# NVML Wrapper ########################################


class NVMLHandle(ctypes.Structure):
    pass


class NVMLMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NVMLUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NVMLDevice:
    def __init__(self, lib, handle):
        self.lib = lib
        self.handle = handle

    def getName(self) -> str:
        name = ctypes.create_string_buffer(96)
        ret = self.lib.nvmlDeviceGetName(self.handle, name, 96)
        assert ret == 0

        return name.value.decode("utf8")

    def getMemoryInfo(self) -> NVMLMemoryInfo:
        info = NVMLMemoryInfo()
        ret = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info))
        assert ret == 0

        return info

    def getUtilizationRates(self) -> NVMLUtilization:
        info = NVMLUtilization()
        ret = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(info))
        assert ret == 0

        return info

    def getTemperature(self) -> int:
        temp = ctypes.c_uint()
        ret = self.lib.nvmlDeviceGetTemperature(
            self.handle, ctypes.c_int(0), ctypes.byref(temp)
        )
        assert ret == 0

        return temp.value

    def getPowerUsage(self) -> int:
        power = ctypes.c_uint()
        ret = self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
        assert ret == 0

        return power.value


class NVML:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")

        self.lib.nvmlInit()

    def deviceGetCount(self):
        num = ctypes.c_uint()
        ret = self.lib.nvmlDeviceGetCount_v2(ctypes.byref(num))
        assert ret == 0

        return num.value

    def device(self, idx: int):
        handle = ctypes.POINTER(NVMLHandle)()
        ret = self.lib.nvmlDeviceGetHandleByIndex(idx, ctypes.byref(handle))
        assert ret == 0

        return NVMLDevice(self.lib, handle)


class Device:
    def __init__(self, nvml_device):
        self.nvml = nvml_device

        self.util = 0.0
        self.power = 0.0

        self.alpha = 1.0 - math.exp(-UPDATE_PERIOD / 2.0)

        name = self.nvml.getName()
        self.name = name.replace("NVIDIA", "").replace("GeForce", "").strip()

        self.update()

    def update(self):
        mem = self.nvml.getMemoryInfo()
        self.used = mem.used / (1024 * 1024 * 1024)
        self.free = mem.total / (1024 * 1024 * 1024)

        util = self.nvml.getUtilizationRates()

        # EMA
        self.util = self.alpha * util.gpu + (1.0 - self.alpha) * self.util

        self.temp = self.nvml.getTemperature()

        power = self.nvml.getPowerUsage()
        self.power = self.alpha * power + (1.0 - self.alpha) * self.power

    def status(self, ic):
        fields = [
            f"{self.name}",
            f"⚙{self.util:3.0f}%",
            f"🛢{self.used:4.1f}/{self.free:.0f} GiB ({self.used / self.free * 100:3.0f}%)",
            f"🌡{self.temp:3}°C",
            f"🔌{self.power / 1000:3.0f}W",
        ]

        return f"[ {'  '.join([f'{ic[0]}{f}{ic[1]}' for f in fields])} ]"


stop_requested = False


def sigint_handler(_, _2):
    global stop_requested

    if stop_requested:
        sys.exit(1)

    stop_requested = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", metavar="PREFIX", help="Display PREFIX in front of the status bar"
    )
    parser.add_argument('command', nargs='*')

    args, command = parser.parse_known_args()
    command = args.command + command

    nvml = NVML()

    num_devices = nvml.deviceGetCount()
    devices = [Device(nvml.device(i)) for i in range(num_devices)]

    stdout_master, stdout_slave = pty.openpty()
    stderr_master, stderr_slave = pty.openpty()

    def update_window_size(*args):
        # Get our window size
        columns, rows = shutil.get_terminal_size()

        # Set window size on the PTYs
        TIOCSWINSZ = getattr(termios, 'TIOCSWINSZ', -2146929561)
        # Note, assume ws_xpixel and ws_ypixel are zero.
        s = struct.pack('HHHH', rows, columns, 0, 0)
        fcntl.ioctl(stdout_master, TIOCSWINSZ, s)
        fcntl.ioctl(stderr_master, TIOCSWINSZ, s)

    update_window_size()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGWINCH, update_window_size)

    proc = subprocess.Popen(
        command, stdout=stdout_slave, stderr=stderr_slave, encoding="utf8"
    )

    os.close(stdout_slave)
    os.close(stderr_slave)

    sys.stdout.write("\n")

    stat_timer = TimerFD(time.CLOCK_MONOTONIC)
    stat_timer.settime(UPDATE_PERIOD, 0.1)

    start_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    running = True

    def status_line():
        current_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

        elapsed = (current_time - start_time) // 1000000000
        elapsed_hours, elapsed = divmod(elapsed, 60 * 60)
        elapsed_minutes, elapsed_seconds = divmod(elapsed, 60)

        ic = ("\033[48;5;18m", "\033[48;5;17m")

        prefix = f"{ic[0]}{args.prefix}{ic[1]} " if args.prefix else ''

        return (
            f"{ic[1]} {prefix}{ic[0]}🕑{elapsed_hours}:{elapsed_minutes:02}:{elapsed_seconds:02}\033{ic[1]} │ "
            + ", ".join([dev.status(ic) for dev in devices])
            + " \033[K\033[0m"
        )

    def handle_output(src, dst):
        try:
            data = os.read(src, 2048)
        except OSError:
            return False

        if not data:
            return False

        # Erase status line and move up
        out = b"\r\033[K\033[1A"

        # Print app data
        out += data

        # Move down again and print status line
        out += b"\n" + status_line().encode()

        dst.buffer.write(out)
        dst.flush()

        return True

    def refresh_status(timer):
        timer.read()

        for dev in devices:
            dev.update()

        sys.stdout.write("\r" + status_line())
        sys.stdout.flush()
        return True

    selector = selectors.DefaultSelector()
    selector.register(
        stdout_master,
        selectors.EVENT_READ,
        data=lambda: handle_output(stdout_master, sys.stdout),
    )
    selector.register(
        stderr_master,
        selectors.EVENT_READ,
        data=lambda: handle_output(stderr_master, sys.stderr),
    )
    selector.register(
        stat_timer, selectors.EVENT_READ, data=lambda: refresh_status(stat_timer)
    )

    running = True
    while running:
        events = selector.select(timeout=1)

        for key, _ in events:
            if not key.data():
                running = False
                break

        if stop_requested:
            proc.send_signal(signal.SIGINT)

    # Go down
    print()

    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
