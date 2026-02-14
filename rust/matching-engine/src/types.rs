pub use ordered_float::OrderedFloat;
use std::fmt;

pub type Price = OrderedFloat<f64>;
pub type Quantity = u64;
pub type OrderId = u64;
pub type Timestamp = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    ImmediateOrCancel,
    FillOrKill,
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: OrderId,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub remaining: Quantity,
    pub order_type: OrderType,
    pub timestamp: Timestamp,
}

impl Order {
    pub fn new(
        id: OrderId,
        side: Side,
        price: f64,
        quantity: Quantity,
        order_type: OrderType,
        timestamp: Timestamp,
    ) -> Self {
        Self {
            id,
            side,
            price: OrderedFloat(price),
            quantity,
            remaining: quantity,
            order_type,
            timestamp,
        }
    }

    pub fn is_filled(&self) -> bool {
        self.remaining == 0
    }
}

#[derive(Debug, Clone)]
pub struct Execution {
    pub buy_order_id: OrderId,
    pub sell_order_id: OrderId,
    pub price: Price,
    pub quantity: Quantity,
    pub timestamp: Timestamp,
}

impl fmt::Display for Execution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trade: buy={} sell={} price={} qty={}",
            self.buy_order_id, self.sell_order_id, self.price, self.quantity
        )
    }
}

#[derive(Debug)]
pub struct L2Snapshot {
    pub bids: Vec<(Price, Quantity)>,
    pub asks: Vec<(Price, Quantity)>,
}
