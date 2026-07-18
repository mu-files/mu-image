# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

# record the time as early as we can to track startup costs
import time as _time
_app_start_time = _time.perf_counter()

 # Start background loading of slow dependencies immediately
from . import deps




