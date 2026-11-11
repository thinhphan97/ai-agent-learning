"""Standalone source-code retrieval fixture."""


class OrderService:
    """Example business logic for a code-aware chunker."""

    def get_paid_orders(self, orders):
        return [order for order in orders if order["status"] == "paid"]

    def calculate_revenue(self, orders):
        paid = self.get_paid_orders(orders)
        return sum(item["amount"] for item in paid)

    def count_paid_orders(self, orders):
        return len(self.get_paid_orders(orders))


def format_currency(value):
    return f"{value:,} VND"


def main():
    service = OrderService()
    revenue = service.calculate_revenue([
        {"status": "paid", "amount": 120000},
        {"status": "pending", "amount": 50000},
    ])
    return format_currency(revenue)
