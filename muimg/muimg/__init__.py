# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

# record the time as early as we can to track startup costs
import time as _time
_app_start_time = _time.perf_counter()

 # Start background loading of slow dependencies immediately
from . import deps




