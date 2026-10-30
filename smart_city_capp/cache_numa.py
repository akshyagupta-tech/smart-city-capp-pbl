"""Compatibility exports for the directory/MESI and mesh simulator API.

The implementation lives in :mod:`memory_numa`; these names are retained for
the original CLI and dashboard imports while exposing the MOESI extensions.
"""

from memory_numa import AdvancedMESICache
from memory_numa import AdvancedMOESICache
from memory_numa import AtomicLLSC
from memory_numa import BrownoutRecoveryJournal
from memory_numa import CacheAccessResult
from memory_numa import CacheAccessStatus
from memory_numa import CircularDMA
from memory_numa import DirectoryEntry
from memory_numa import DirectoryState
from memory_numa import ECCCheckResult
from memory_numa import HardwareLockSync
from memory_numa import HardwareBarrierUnit
from memory_numa import LoadLinkedStoreConditional
from memory_numa import MeshInterconnect
from memory_numa import MeshNode
from memory_numa import NVMJournalEntry
from memory_numa import NUMADirectorySystem
from memory_numa import NonBlockingCache
from memory_numa import RoutedMessage
from memory_numa import SECDEDECC
from memory_numa import SECDEDWord
from memory_numa import VirtualChannel
from types_def import MOESIState

__all__ = [
    "AdvancedMESICache",
    "AdvancedMOESICache",
    "AtomicLLSC",
    "BrownoutRecoveryJournal",
    "CacheAccessResult",
    "CacheAccessStatus",
    "CircularDMA",
    "DirectoryEntry",
    "DirectoryState",
    "ECCCheckResult",
    "HardwareLockSync",
    "HardwareBarrierUnit",
    "LoadLinkedStoreConditional",
    "MeshInterconnect",
    "MeshNode",
    "MOESIState",
    "NVMJournalEntry",
    "NUMADirectorySystem",
    "NonBlockingCache",
    "RoutedMessage",
    "SECDEDECC",
    "SECDEDWord",
    "VirtualChannel",
]
