use std::collections::{BTreeMap, VecDeque};

use crate::types::*;

pub struct OrderBook {
    bids: BTreeMap<Price, VecDeque<Order>>,
    asks: BTreeMap<Price, VecDeque<Order>>,
    orders: std::collections::HashMap<OrderId, (Side, Price)>,
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: std::collections::HashMap::new(),
        }
    }

    pub fn submit(&mut self, order: Order, timestamp: Timestamp) -> Vec<Execution> {
        match order.order_type {
            OrderType::Limit => self.process_limit(order, timestamp),
            OrderType::Market => self.process_market(order, timestamp),
            OrderType::ImmediateOrCancel => self.process_ioc(order, timestamp),
            OrderType::FillOrKill => self.process_fok(order, timestamp),
        }
    }

    pub fn cancel(&mut self, order_id: OrderId) -> bool {
        if let Some((side, price)) = self.orders.remove(&order_id) {
            let book = match side {
                Side::Buy => &mut self.bids,
                Side::Sell => &mut self.asks,
            };
            if let Some(level) = book.get_mut(&price) {
                level.retain(|o| o.id != order_id);
                if level.is_empty() {
                    book.remove(&price);
                }
            }
            true
        } else {
            false
        }
    }

    pub fn best_bid(&self) -> Option<Price> {
        self.bids.keys().next_back().copied()
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks.keys().next().copied()
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((ask - bid).into_inner()),
            _ => None,
        }
    }

    pub fn snapshot(&self, depth: usize) -> L2Snapshot {
        let bids: Vec<(Price, Quantity)> = self
            .bids
            .iter()
            .rev()
            .take(depth)
            .map(|(&p, q)| (p, q.iter().map(|o| o.remaining).sum()))
            .collect();

        let asks: Vec<(Price, Quantity)> = self
            .asks
            .iter()
            .take(depth)
            .map(|(&p, q)| (p, q.iter().map(|o| o.remaining).sum()))
            .collect();

        L2Snapshot { bids, asks }
    }

    fn process_limit(&mut self, mut order: Order, timestamp: Timestamp) -> Vec<Execution> {
        let executions = self.try_match(&mut order, timestamp);
        if order.remaining > 0 {
            self.add_to_book(order);
        }
        executions
    }

    fn process_market(&mut self, mut order: Order, timestamp: Timestamp) -> Vec<Execution> {
        self.try_match(&mut order, timestamp)
    }

    fn process_ioc(&mut self, mut order: Order, timestamp: Timestamp) -> Vec<Execution> {
        self.try_match(&mut order, timestamp)
    }

    fn process_fok(&mut self, order: Order, timestamp: Timestamp) -> Vec<Execution> {
        let available = self.available_quantity(&order);
        if available >= order.remaining {
            let mut order = order;
            self.try_match(&mut order, timestamp)
        } else {
            vec![]
        }
    }

    fn available_quantity(&self, order: &Order) -> Quantity {
        let book = match order.side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };

        let mut total = 0u64;
        for (&price, level) in book.iter() {
            let crosses = match order.side {
                Side::Buy => price <= order.price,
                Side::Sell => price >= order.price,
            };
            if !crosses && order.order_type != OrderType::Market {
                break;
            }
            total += level.iter().map(|o| o.remaining).sum::<u64>();
        }
        total
    }

    fn try_match(&mut self, order: &mut Order, timestamp: Timestamp) -> Vec<Execution> {
        let mut executions = Vec::new();
        let opposite = match order.side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };

        let mut empty_levels = Vec::new();

        let iter: Vec<Price> = match order.side {
            Side::Buy => opposite.keys().copied().collect(),
            Side::Sell => opposite.keys().rev().copied().collect(),
        };

        for price in iter {
            if order.remaining == 0 {
                break;
            }

            let crosses = match order.side {
                Side::Buy => price <= order.price || order.order_type == OrderType::Market,
                Side::Sell => price >= order.price || order.order_type == OrderType::Market,
            };
            if !crosses {
                break;
            }

            if let Some(level) = opposite.get_mut(&price) {
                while order.remaining > 0 && !level.is_empty() {
                    let resting = level.front_mut().unwrap();
                    let fill_qty = order.remaining.min(resting.remaining);

                    order.remaining -= fill_qty;
                    resting.remaining -= fill_qty;

                    let (buy_id, sell_id) = match order.side {
                        Side::Buy => (order.id, resting.id),
                        Side::Sell => (resting.id, order.id),
                    };

                    executions.push(Execution {
                        buy_order_id: buy_id,
                        sell_order_id: sell_id,
                        price,
                        quantity: fill_qty,
                        timestamp,
                    });

                    if resting.is_filled() {
                        let filled = level.pop_front().unwrap();
                        self.orders.remove(&filled.id);
                    }
                }
                if level.is_empty() {
                    empty_levels.push(price);
                }
            }
        }

        let opposite = match order.side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };
        for price in empty_levels {
            opposite.remove(&price);
        }

        executions
    }

    fn add_to_book(&mut self, order: Order) {
        let book = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        self.orders.insert(order.id, (order.side, order.price));
        book.entry(order.price).or_default().push_back(order);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limit_buy(id: u64, price: f64, qty: u64) -> Order {
        Order::new(id, Side::Buy, price, qty, OrderType::Limit, 0)
    }

    fn limit_sell(id: u64, price: f64, qty: u64) -> Order {
        Order::new(id, Side::Sell, price, qty, OrderType::Limit, 0)
    }

    #[test]
    fn test_limit_order_no_match() {
        let mut book = OrderBook::new();
        let execs = book.submit(limit_buy(1, 100.0, 10), 1);
        assert!(execs.is_empty());
        assert_eq!(book.best_bid(), Some(OrderedFloat(100.0)));
    }

    #[test]
    fn test_limit_order_exact_match() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 10), 1);
        let execs = book.submit(limit_buy(2, 100.0, 10), 2);
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].quantity, 10);
        assert_eq!(execs[0].price, OrderedFloat(100.0));
    }

    #[test]
    fn test_partial_fill() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 10), 1);
        let execs = book.submit(limit_buy(2, 100.0, 5), 2);
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].quantity, 5);
        assert_eq!(book.best_ask(), Some(OrderedFloat(100.0)));
    }

    #[test]
    fn test_price_time_priority() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 101.0, 10), 1);
        book.submit(limit_sell(2, 100.0, 5), 2);
        book.submit(limit_sell(3, 100.0, 5), 3);

        let execs = book.submit(limit_buy(4, 101.0, 8), 4);
        assert_eq!(execs[0].price, OrderedFloat(100.0));
        assert_eq!(execs[0].sell_order_id, 2);
    }

    #[test]
    fn test_market_order() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 5), 1);
        book.submit(limit_sell(2, 101.0, 5), 2);

        let market = Order::new(3, Side::Buy, 0.0, 8, OrderType::Market, 3);
        let execs = book.submit(market, 3);
        assert_eq!(execs.len(), 2);
        assert_eq!(execs[0].quantity, 5);
        assert_eq!(execs[1].quantity, 3);
    }

    #[test]
    fn test_cancel_order() {
        let mut book = OrderBook::new();
        book.submit(limit_buy(1, 100.0, 10), 1);
        assert!(book.cancel(1));
        assert_eq!(book.best_bid(), None);
        assert!(!book.cancel(1));
    }

    #[test]
    fn test_ioc_partial_fill() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 5), 1);

        let ioc = Order::new(2, Side::Buy, 100.0, 10, OrderType::ImmediateOrCancel, 2);
        let execs = book.submit(ioc, 2);
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].quantity, 5);
        assert_eq!(book.best_bid(), None);
    }

    #[test]
    fn test_fok_rejected() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 5), 1);

        let fok = Order::new(2, Side::Buy, 100.0, 10, OrderType::FillOrKill, 2);
        let execs = book.submit(fok, 2);
        assert!(execs.is_empty());
        assert_eq!(book.best_ask(), Some(OrderedFloat(100.0)));
    }

    #[test]
    fn test_spread() {
        let mut book = OrderBook::new();
        book.submit(limit_buy(1, 99.0, 10), 1);
        book.submit(limit_sell(2, 101.0, 10), 2);
        assert_eq!(book.spread(), Some(2.0));
    }

    #[test]
    fn test_l2_snapshot() {
        let mut book = OrderBook::new();
        book.submit(limit_buy(1, 99.0, 10), 1);
        book.submit(limit_buy(2, 98.0, 20), 2);
        book.submit(limit_sell(3, 101.0, 15), 3);

        let snap = book.snapshot(5);
        assert_eq!(snap.bids.len(), 2);
        assert_eq!(snap.asks.len(), 1);
        assert_eq!(snap.bids[0].0, OrderedFloat(99.0));
    }
}
