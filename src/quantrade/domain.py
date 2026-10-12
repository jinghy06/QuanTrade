from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str = ""
    theme: str = "unknown"
    listing_date: str = ""
    lot_size: int = 100
    t_plus: int = 1
    verified: bool = False


@dataclass
class Target:
    signal_time: str
    weights: dict[str, float]
    reasons: list[str]
    valid_until: str
    model_version: str | None = None
    factors: dict = field(default_factory=dict)
    strategy_version: str = "3.0.0"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    date: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    source: str = "simulation"


@dataclass
class Position:
    quantity: float = 0.0
    average_cost: float = 0.0
    last_price: float = 0.0
    price_date: str = ""
    # Each tuple is (trade date of purchase, still locked quantity).
    locked: list = field(default_factory=list)
