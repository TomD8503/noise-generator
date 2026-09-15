# project_class.py

import pickle
import zlib
import binascii
import time
from dataclasses import dataclass, field
import numpy as np
import math
from numpy.fft import irfft


@dataclass
class Project:
    # -------------------------
    # 1. METADATA
    # -------------------------
    format_version: int = 1
    last_modified: float = 0.0
    notes: str = ""

    # -------------------------
    # 2. CONFIGURATION (explicit fields)
    # -------------------------
    sample_rate: int = 48000
    fft_size: int = 65536
    duration_seconds: float = 10.0

    freq_low: float = 20.0
    freq_high: float = 20000.0
    noise_color: str = "pink"

    phase_levels: int = 16384
    phase_step: float = 0.0  # will be computed

    target_crest_db: float = 5.0

    octaves: int = 3

    # -------------------------
    # 3. BEST CANDIDATE
    # -------------------------
    best_error: float = float("inf")
    best_crest_db: float = float("inf")
    best_phases_uint: np.ndarray = None

    # -------------------------
    # 4. CACHED DATA (initialized as None)
    # -------------------------
    _freqs: np.ndarray = field(default=None, repr=False)
    _magnitude_mask: np.ndarray = field(default=None, repr=False)
    _phase_lut: np.ndarray = field(default=None, repr=False)
    _active_idx: np.ndarray = field(default=None, repr=False)

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def initialize(self):
        """Compute derived config values."""
        self.phase_step = 2 * np.pi / self.phase_levels
        self.last_modified = time.time()
        # Clear cache when config changes
        self._clear_cache()

    def _clear_cache(self):
        """Clear cached data when config changes."""
        self._freqs = None
        self._magnitude_mask = None
        self._phase_lut = None
        self._active_idx = None

    def _ensure_initialized(self):
        """Make sure all cached data is generated."""
        if self._freqs is None:
            self._generate_freqs()
        if self._phase_lut is None:
            self._generate_phase_lut()
        if self._magnitude_mask is None:
            self._generate_magnitude_mask()

    def _generate_freqs(self):
        """Generate frequency array (positive frequencies only)."""
        self._freqs = np.fft.rfftfreq(self.fft_size, d=1/self.sample_rate)
        print(f"✓ Frequencies generated: {len(self._freqs)} bins")

    def _generate_phase_lut(self):
        """Generate phase lookup table."""
        self._phase_lut = np.exp(
            1j * np.arange(self.phase_levels, dtype=np.float32) * self.phase_step
        ).astype(np.complex64)
        print(f"✓ Phase LUT generated: {len(self._phase_lut)} entries")

    def _generate_magnitude_mask(self):
        """Generate magnitude mask based on current config."""
        if self._freqs is None:
            self._generate_freqs()

        magnitudes = np.zeros_like(self._freqs, dtype=np.float64)

        f = self._freqs.copy()
        f[f == 0] = 1e-12  # avoid log(0)

        # Convert to log2 domain
        x = np.log2(f)
        x_lo = np.log2(self.freq_low)
        x_hi = np.log2(self.freq_high)

        # Taper width in octaves
        T = 1.0 / self.octaves

        # Flat passband
        passband = (x >= x_lo) & (x <= x_hi)
        magnitudes[passband] = 1.0

        # Lower taper
        low_taper = (x >= x_lo - T) & (x < x_lo)
        if np.any(low_taper):
            t = (x[low_taper] - (x_lo - T)) / T
            magnitudes[low_taper] = 0.5 * (1 - np.cos(np.pi * t))

        # Upper taper
        high_taper = (x > x_hi) & (x <= x_hi + T)
        if np.any(high_taper):
            t = ((x_hi + T) - x[high_taper]) / T
            magnitudes[high_taper] = 0.5 * (1 - np.cos(np.pi * t))

        # Pink/white weighting
        if self.noise_color == "pink":
            pink_amp = 1.0 / np.sqrt(f)
            pink_amp[0] = 0
            magnitudes *= pink_amp
        elif self.noise_color != "white":
            raise ValueError("Unsupported noise_color")

        self._magnitude_mask = magnitudes
        self._active_idx = np.where(magnitudes > 0)[0]
        print(f"✓ Magnitude mask generated: {len(self._active_idx)} active bins")

    # ============================================================
    # PUBLIC METHODS
    # ============================================================
    def generate_phases_uint(self):
        """Generate phases (0 or π) using Rudin–Shapiro."""
        self._ensure_initialized()
        N = len(self._freqs)

        # Rudin–Shapiro generator ±1 → 0/π
        n = np.arange(N)
        x = n & (n >> 1)
        parity = np.zeros(N, dtype=np.uint8)
        temp = x.copy()
        while np.any(temp):
            parity ^= 1
            temp &= temp - 1

        # zamieniamy 0 -> 0, 1 -> π
        phases_uint = parity.astype(np.uint16) * np.uint16(np.pi)

        return phases_uint

    def generate_waveform(self, phases_uint):
        self._ensure_initialized()
        # Synthesize waveform
        spectrum = np.zeros(len(self._freqs), dtype=np.complex64)
        spectrum[self._active_idx] = (
            self._magnitude_mask[self._active_idx] * 
            self._phase_lut[phases_uint[self._active_idx]]
        )
        spectrum[0] = spectrum[0].real + 0j
        if self.fft_size % 2 == 0:
            spectrum[-1] = spectrum[-1].real + 0j
        
        waveform = irfft(spectrum, n=self.fft_size)    
        return waveform

    def generate_and_evaluate(self):
        phases = self.generate_phases_uint()
        waveform = self.generate_waveform(phases)
        cf_db = self.crest_factor(waveform)
        error = abs(cf_db - self.target_crest_db)
        return waveform, phases, error, cf_db

    def is_new_best(self, phases_uint, error, crest_db):
        if error < self.best_error:
            self.best_error = error
            self.best_crest_db = crest_db
            self.best_phases_uint = phases_uint.astype(np.uint16)
            self.last_modified = time.time()
            return True
        return False

    def crest_factor(self, period):
        rms = np.sqrt(np.mean(period**2))
        peak = np.max(np.abs(period))
        if rms == 0 or peak == 0:
            return np.inf
        cf = peak / rms
        return 20 * np.log10(cf)

    def get_info(self):
        """Get info about current state."""
        self._ensure_initialized()
        
        return {
            'config': {
                'sample_rate': self.sample_rate,
                'fft_size': self.fft_size,
                'freq_range': [self.freq_low, self.freq_high],
                'noise_color': self.noise_color,
                'phase_levels': self.phase_levels,
                'target_crest_db': self.target_crest_db,
                'octaves': self.octaves
            },
            'cached': {
                'freq_bins': len(self._freqs) if self._freqs is not None else 0,
                'active_bins': len(self._active_idx) if self._active_idx is not None else 0,
                'phase_lut_size': len(self._phase_lut) if self._phase_lut is not None else 0
            },
            'best': {
                'error': self.best_error,
                'crest_db': self.best_crest_db,
                'has_candidate': self.best_phases_uint is not None
            }
        }

    # ============================================================
    # SAVE / LOAD
    # ============================================================

    def save(self, path):
        """Save project to file."""
        # Don't save cached data
        freqs = self._freqs
        mag_mask = self._magnitude_mask
        phase_lut = self._phase_lut
        active_idx = self._active_idx
        
        self._freqs = None
        self._magnitude_mask = None
        self._phase_lut = None
        self._active_idx = None
        
        raw = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(raw, level=9)
        crc = binascii.crc32(compressed)
        
        with open(path, "wb") as f:
            f.write(crc.to_bytes(4, "big"))
            f.write(compressed)
        
        # Restore cache
        self._freqs = freqs
        self._magnitude_mask = mag_mask
        self._phase_lut = phase_lut
        self._active_idx = active_idx

    @staticmethod
    def load(path):
        """Load project from file."""
        with open(path, "rb") as f:
            crc_stored = int.from_bytes(f.read(4), "big")
            compressed = f.read()
        
        if binascii.crc32(compressed) != crc_stored:
            raise ValueError("CRC mismatch — project file corrupted")
        
        raw = zlib.decompress(compressed)
        return pickle.loads(raw)
