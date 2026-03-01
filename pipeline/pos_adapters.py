"""
pipeline/pos_adapters.py
------------------------
Adapter design pattern for normalizing disparate Tier-1 POS webhook payloads
into a single canonical ZomatoKPTSchema before the signal passes to the main
pipeline.

Two concrete adapters are implemented:
  - PetpoojaAdapter  (Petpooja POS webhook format)
  - PosistAdapter    (Posist / POSist webhook format)

Both implement the POSGateway interface and output a ZomatoKPTSchema dict.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Canonical output schema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZomatoKPTSchema:
    """
    Canonical representation of a kitchen-ready signal after POS normalisation.
    This is what every adapter must produce before handing off to the pipeline.
    """
    restaurant_id: str
    order_id: str
    ticket_cleared_at: datetime        # when the KOT was closed — ground truth
    order_placed_at: datetime          # when the order was created in the POS
    item_count: int
    is_delivery_order: bool
    pos_source: str                    # "petpooja" | "posist" | etc.
    raw_payload: dict = field(default_factory=dict, repr=False)

    @property
    def pos_kpt_minutes(self) -> float:
        """
        Kitchen Preparation Time as measured purely from POS timestamps.
        This is the clean, human-independent signal used for Tier 1 merchants.
        """
        delta = self.ticket_cleared_at - self.order_placed_at
        return round(delta.total_seconds() / 60, 2)

    def to_dict(self) -> dict:
        return {
            "restaurant_id": self.restaurant_id,
            "order_id": self.order_id,
            "ticket_cleared_at": self.ticket_cleared_at.isoformat(),
            "order_placed_at": self.order_placed_at.isoformat(),
            "item_count": self.item_count,
            "is_delivery_order": self.is_delivery_order,
            "pos_source": self.pos_source,
            "pos_kpt_minutes": self.pos_kpt_minutes,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Abstract gateway interface
# ──────────────────────────────────────────────────────────────────────────────

class POSGateway(abc.ABC):
    """
    Abstract base class (interface) for all POS adapters.

    Any new POS integration (DotPe, Torqus, etc.) must subclass this and
    implement `normalize`. The rest of the pipeline never touches raw POS
    payloads directly.
    """

    @abc.abstractmethod
    def normalize(self, raw_payload: dict[str, Any]) -> ZomatoKPTSchema:
        """
        Accept a raw webhook payload in the vendor's native format and return
        a normalised ZomatoKPTSchema instance.

        Raises:
            KeyError:   if a required field is missing from the payload.
            ValueError: if a timestamp cannot be parsed.
        """

    @staticmethod
    def _parse_dt(value: Any, fmt: Optional[str] = None) -> datetime:
        """
        Helper: parse a datetime from either a string with a known format,
        an ISO-8601 string, or a Unix timestamp (int/float).
        """
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            if fmt:
                return datetime.strptime(value, fmt)
            # Try ISO-8601 first, then a few common Indian formats
            for attempt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                            "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
                try:
                    return datetime.strptime(value, attempt)
                except ValueError:
                    continue
        raise ValueError(f"Cannot parse datetime from: {value!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Concrete adapter: Petpooja
# ──────────────────────────────────────────────────────────────────────────────

class PetpoojaAdapter(POSGateway):
    """
    Normalises a Petpooja webhook payload into ZomatoKPTSchema.

    Expected Petpooja payload structure (mock):
    {
        "restID":       "PP_R042",
        "billNo":       "BL-20250101-0081",
        "createdOn":    "01/01/2025 13:22:00",        # DD/MM/YYYY HH:MM:SS
        "kotClearedOn": "01/01/2025 13:38:45",
        "itemCount":    3,
        "orderType":    "delivery"                    # "delivery" | "dine-in"
    }
    """

    POS_SOURCE = "petpooja"

    def normalize(self, raw_payload: dict[str, Any]) -> ZomatoKPTSchema:
        try:
            return ZomatoKPTSchema(
                restaurant_id=str(raw_payload["restID"]),
                order_id=str(raw_payload["billNo"]),
                order_placed_at=self._parse_dt(
                    raw_payload["createdOn"], fmt="%d/%m/%Y %H:%M:%S"
                ),
                ticket_cleared_at=self._parse_dt(
                    raw_payload["kotClearedOn"], fmt="%d/%m/%Y %H:%M:%S"
                ),
                item_count=int(raw_payload["itemCount"]),
                is_delivery_order=(
                    raw_payload.get("orderType", "").lower() == "delivery"
                ),
                pos_source=self.POS_SOURCE,
                raw_payload=raw_payload,
            )
        except KeyError as exc:
            raise KeyError(
                f"PetpoojaAdapter: missing required field {exc} in payload"
            ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Concrete adapter: Posist
# ──────────────────────────────────────────────────────────────────────────────

class PosistAdapter(POSGateway):
    """
    Normalises a Posist (POSist) webhook payload into ZomatoKPTSchema.

    Posist uses Unix timestamps and a different field naming convention:

    {
        "outlet_ref_id":     "POSIST_R017",
        "transaction_id":    "TXN-88821",
        "order_time_epoch":  1735731720,        # Unix timestamp (seconds)
        "kot_close_epoch":   1735732710,
        "no_of_items":       5,
        "channel":           "zomato"           # "zomato" | "swiggy" | "dine_in"
    }
    """

    POS_SOURCE = "posist"
    DELIVERY_CHANNELS = {"zomato", "swiggy", "online", "delivery"}

    def normalize(self, raw_payload: dict[str, Any]) -> ZomatoKPTSchema:
        try:
            return ZomatoKPTSchema(
                restaurant_id=str(raw_payload["outlet_ref_id"]),
                order_id=str(raw_payload["transaction_id"]),
                order_placed_at=self._parse_dt(raw_payload["order_time_epoch"]),
                ticket_cleared_at=self._parse_dt(raw_payload["kot_close_epoch"]),
                item_count=int(raw_payload["no_of_items"]),
                is_delivery_order=(
                    raw_payload.get("channel", "").lower()
                    in self.DELIVERY_CHANNELS
                ),
                pos_source=self.POS_SOURCE,
                raw_payload=raw_payload,
            )
        except KeyError as exc:
            raise KeyError(
                f"PosistAdapter: missing required field {exc} in payload"
            ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────

_ADAPTER_REGISTRY: dict[str, type[POSGateway]] = {
    "petpooja": PetpoojaAdapter,
    "posist":   PosistAdapter,
}


def get_adapter(pos_vendor: str) -> POSGateway:
    """
    Factory: returns the correct adapter instance for a given POS vendor name.

    Usage:
        adapter = get_adapter("petpooja")
        schema  = adapter.normalize(raw_webhook_payload)
    """
    vendor = pos_vendor.lower().strip()
    if vendor not in _ADAPTER_REGISTRY:
        supported = list(_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"No adapter registered for POS vendor '{pos_vendor}'. "
            f"Supported: {supported}"
        )
    return _ADAPTER_REGISTRY[vendor]()


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test — run this file directly to verify both adapters work
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    petpooja_payload = {
        "restID":       "PP_R042",
        "billNo":       "BL-20250101-0081",
        "createdOn":    "01/01/2025 13:22:00",
        "kotClearedOn": "01/01/2025 13:38:45",
        "itemCount":    3,
        "orderType":    "delivery",
    }

    posist_payload = {
        "outlet_ref_id":    "POSIST_R017",
        "transaction_id":   "TXN-88821",
        "order_time_epoch": 1735731720,
        "kot_close_epoch":  1735732710,
        "no_of_items":      5,
        "channel":          "zomato",
    }

    for vendor, payload in [("petpooja", petpooja_payload),
                             ("posist",  posist_payload)]:
        adapter = get_adapter(vendor)
        schema  = adapter.normalize(payload)
        print(f"\n[{vendor.upper()}] Normalised schema:")
        for k, v in schema.to_dict().items():
            print(f"  {k:<22} = {v}")
