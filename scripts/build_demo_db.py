"""Build data/demo.db — a small synthetic retail database for TableTalk.

"Acme Outfitters" sells outdoor gear through web, mobile, and store
channels. The schema is small enough to read in one sitting but has the
features that make text-to-SQL hard in practice: multi-hop foreign keys,
categorical status columns, money stored per line item (not per order),
nullable FKs, and dates that need range filters.

    regions ─< customers ─< orders >─ sales_reps >─ regions
                    │          │
                    │          └─< order_items >─ products >─ categories
                    └─< support_tickets >─ orders (nullable)

The data is generated from a fixed random seed, so every build is
byte-for-byte identical and the tests and golden set can depend on it.

Run:
    python scripts/build_demo_db.py            # writes data/demo.db
    python scripts/build_demo_db.py out.db     # custom path
"""

import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "demo.db"

SEED = 20260101
START = date(2024, 1, 1)
END = date(2026, 9, 30)  # last order date; "today" for the dataset

SCHEMA = """
CREATE TABLE regions (
    region_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    country     TEXT NOT NULL
);

CREATE TABLE sales_reps (
    rep_id      INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    region_id   INTEGER NOT NULL REFERENCES regions(region_id),
    hire_date   DATE NOT NULL
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    city        TEXT NOT NULL,
    region_id   INTEGER NOT NULL REFERENCES regions(region_id),
    segment     TEXT NOT NULL,          -- Consumer | Small Business | Enterprise
    signup_date DATE NOT NULL
);

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE products (
    product_id  INTEGER PRIMARY KEY,
    sku         TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(category_id),
    list_price  REAL NOT NULL,
    is_active   INTEGER NOT NULL        -- 1 = currently sold, 0 = discontinued
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    rep_id      INTEGER REFERENCES sales_reps(rep_id),  -- NULL for self-serve web/mobile orders
    order_date  DATE NOT NULL,
    channel     TEXT NOT NULL,          -- web | mobile | store
    status      TEXT NOT NULL           -- delivered | shipped | processing | cancelled | returned
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL,        -- price actually charged, may differ from list_price
    discount_pct  REAL NOT NULL         -- 0..0.3, applied to quantity * unit_price
);

CREATE TABLE support_tickets (
    ticket_id   INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_id    INTEGER REFERENCES orders(order_id),  -- NULL for general inquiries
    opened_at   DATETIME NOT NULL,
    closed_at   DATETIME,               -- NULL while open
    priority    TEXT NOT NULL,          -- low | medium | high | urgent
    status      TEXT NOT NULL,          -- open | pending | resolved
    topic       TEXT NOT NULL           -- shipping | billing | product defect | return | account
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
"""

REGIONS = [
    ("Pacific Northwest", "USA"), ("California", "USA"), ("Mountain West", "USA"),
    ("Midwest", "USA"), ("Northeast", "USA"), ("Southeast", "USA"),
    ("Ontario", "Canada"), ("British Columbia", "Canada"),
]
CITIES = {
    "Pacific Northwest": ["Seattle", "Portland", "Spokane", "Tacoma"],
    "California": ["San Francisco", "Los Angeles", "San Diego", "Sacramento"],
    "Mountain West": ["Denver", "Boulder", "Salt Lake City", "Boise"],
    "Midwest": ["Chicago", "Minneapolis", "Madison", "Columbus"],
    "Northeast": ["Boston", "Burlington", "New York", "Portland ME"],
    "Southeast": ["Atlanta", "Asheville", "Charlotte", "Nashville"],
    "Ontario": ["Toronto", "Ottawa", "Hamilton"],
    "British Columbia": ["Vancouver", "Victoria", "Whistler"],
}
FIRST = ["Ava", "Liam", "Maya", "Noah", "Zoe", "Ethan", "Priya", "Lucas", "Chloe",
         "Mateo", "Hana", "Owen", "Isla", "Arjun", "Nora", "Felix", "Lena", "Omar",
         "Ruby", "Kai", "Elena", "Theo", "Aisha", "Jonah", "Mila", "Diego", "Sofia",
         "Ravi", "Grace", "Leo"]
LAST = ["Nguyen", "Smith", "Patel", "Garcia", "Kim", "Johnson", "Silva", "Chen",
        "Brown", "Rossi", "Okafor", "Martin", "Haddad", "Lopez", "Novak", "Wilson",
        "Tanaka", "Murphy", "Dubois", "Singh", "Clark", "Larsen", "Moreau", "Reyes"]
SEGMENTS = [("Consumer", 0.72), ("Small Business", 0.22), ("Enterprise", 0.06)]

CATALOG = {
    "Tents": [("Ridgeline 2P Tent", 289), ("Basecamp 4P Tent", 449), ("Ultralight 1P Tent", 379),
              ("Family Cabin 6P Tent", 599)],
    "Sleeping Bags": [("Summit -10C Down Bag", 419), ("Trail 0C Synthetic Bag", 149),
                      ("Kids Explorer Bag", 69), ("Quilt 5C Down", 259)],
    "Backpacks": [("Trailhead 28L Daypack", 119), ("Expedition 65L Pack", 299),
                  ("Commuter 22L Pack", 89), ("Hydration 12L Vest", 99), ("Thru-Hiker 50L Pack", 249)],
    "Footwear": [("Granite Mid GTX Boot", 219), ("Trail Runner 5", 149), ("Camp Sandal", 59),
                 ("Alpine Mountaineering Boot", 389), ("Approach Shoe", 159)],
    "Apparel": [("Storm Shell Jacket", 279), ("Merino Base Layer Top", 89), ("Fleece Midlayer", 109),
                ("Hiking Pants", 79), ("Down Puffy Jacket", 229), ("Sun Hoodie", 59), ("Wool Socks 3-Pack", 39)],
    "Cooking": [("Titan Backpacking Stove", 79), ("Nested Cook Set", 69), ("Insulated Mug", 29),
                ("Water Filter Pro", 99), ("Camp Kitchen Table", 139)],
    "Climbing": [("Dynamic Rope 60m", 229), ("Harness All-Round", 79), ("Locking Carabiner", 15),
                 ("Climbing Helmet", 89), ("Chalk Bag", 25), ("Belay Device", 35)],
    "Navigation & Electronics": [("GPS Watch Trail", 349), ("Headlamp 400lm", 49), ("Solar Charger 20W", 89),
                                 ("Satellite Messenger", 299), ("Compass Classic", 39)],
    "Bikes & Accessories": [("Gravel Bike Frame Bag", 69), ("Trail Bike Helmet", 129), ("Bike Light Set", 59),
                            ("Repair Kit", 35)],
}
DISCONTINUED = {"Kids Explorer Bag", "Commuter 22L Pack", "Camp Sandal", "Compass Classic"}

CHANNELS = [("web", 0.5), ("mobile", 0.3), ("store", 0.2)]
STATUS_RECENT = [("processing", 0.55), ("shipped", 0.45)]
STATUS_OLD = [("delivered", 0.86), ("cancelled", 0.06), ("returned", 0.08)]
TOPICS = ["shipping", "billing", "product defect", "return", "account"]
PRIORITIES = [("low", 0.3), ("medium", 0.4), ("high", 0.22), ("urgent", 0.08)]


def _pick(rng: random.Random, weighted: list[tuple]) -> str:
    return rng.choices([v for v, _ in weighted], weights=[w for _, w in weighted])[0]


def _day(rng: random.Random, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def _seasonal_day(rng: random.Random) -> date:
    """Order dates with growth over time and a Q4/summer bump, so trend
    questions have a real answer."""
    while True:
        d = _day(rng, START, END)
        growth = 0.6 + 0.4 * (d - START).days / (END - START).days
        season = 1.35 if d.month in (6, 7, 11, 12) else 1.0
        if rng.random() < growth * season / 1.35:
            return d


def build(path: Path) -> dict:
    rng = random.Random(SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    conn.executemany("INSERT INTO regions VALUES (?, ?, ?)",
                     [(i, n, c) for i, (n, c) in enumerate(REGIONS, 1)])
    region_names = {i: n for i, (n, _) in enumerate(REGIONS, 1)}

    reps = []
    for rep_id in range(1, 17):
        region_id = (rep_id - 1) % len(REGIONS) + 1
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        reps.append((rep_id, name, region_id, _day(rng, date(2018, 1, 1), date(2025, 6, 1)).isoformat()))
    conn.executemany("INSERT INTO sales_reps VALUES (?, ?, ?, ?)", reps)
    reps_by_region: dict[int, list[int]] = {}
    for rep_id, _, region_id, _ in reps:
        reps_by_region.setdefault(region_id, []).append(rep_id)

    customers = []
    used_emails = set()
    for cid in range(1, 601):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        email = f"{first}.{last}{cid}@example.com".lower()
        used_emails.add(email)
        region_id = rng.choices(range(1, len(REGIONS) + 1), weights=[14, 18, 12, 14, 15, 12, 9, 6])[0]
        city = rng.choice(CITIES[region_names[region_id]])
        customers.append((cid, f"{first} {last}", email, city, region_id,
                          _pick(rng, SEGMENTS), _day(rng, date(2022, 1, 1), date(2025, 12, 31)).isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", customers)

    products = []
    pid = 0
    for cat_id, (cat, items) in enumerate(CATALOG.items(), 1):
        conn.execute("INSERT INTO categories VALUES (?, ?)", (cat_id, cat))
        for name, price in items:
            pid += 1
            sku = f"{cat[:3].upper()}-{pid:04d}"
            products.append((pid, sku, name, cat_id, float(price), 0 if name in DISCONTINUED else 1))
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", products)

    # Popularity weights: cheap consumables sell more often.
    weights = [1 / (p[4] ** 0.5) for p in products]

    orders, items, tickets = [], [], []
    item_id = 0
    recent_cutoff = END - timedelta(days=10)
    # A long-tail of activity: a few customers order a lot, many order rarely.
    cust_weights = [min(rng.paretovariate(1.3), 12.0) for _ in customers]
    for oid in range(1, 4801):
        d = _seasonal_day(rng)
        # Only customers who had signed up by the order date can place it.
        c = rng.choices(customers, weights=cust_weights)[0]
        while date.fromisoformat(c[6]) > d:
            c = rng.choices(customers, weights=cust_weights)[0]
        cid, region_id, segment = c[0], c[4], c[5]
        channel = _pick(rng, CHANNELS)
        if segment != "Consumer" and rng.random() < 0.5:
            channel = "store"
        rep_id = rng.choice(reps_by_region[region_id]) if channel == "store" or segment == "Enterprise" else None
        status = _pick(rng, STATUS_RECENT if d > recent_cutoff else STATUS_OLD)
        orders.append((oid, cid, rep_id, d.isoformat(), channel, status))

        n_lines = rng.choices([1, 2, 3, 4, 5], weights=[40, 28, 17, 10, 5])[0]
        for prod in rng.sample(rng.choices(products, weights=weights, k=12), k=n_lines):
            if not prod[5] and d > date(2025, 6, 30):
                continue  # discontinued products stopped selling mid-2025
            item_id += 1
            qty = rng.choices([1, 2, 3, 4, 6, 10], weights=[60, 20, 8, 5, 4, 3])[0]
            if segment == "Enterprise":
                qty *= rng.choice([2, 3, 5])
            price = round(prod[4] * rng.choice([1.0, 1.0, 1.0, 0.95, 0.9, 1.05]), 2)
            discount = rng.choice([0.0] * 6 + [0.05, 0.1, 0.15, 0.2, 0.3]) if segment != "Consumer" \
                else rng.choice([0.0] * 10 + [0.1, 0.2])
            items.append((item_id, oid, prod[0], qty, price, discount))

        if status in ("returned", "cancelled") or rng.random() < 0.06:
            topic = "return" if status == "returned" else rng.choice(TOPICS)
            opened = datetime.combine(d, datetime.min.time()) + timedelta(
                days=rng.randint(1, 20), hours=rng.randint(8, 18), minutes=rng.randint(0, 59))
            if opened.date() > END:
                opened = datetime.combine(END, datetime.min.time()) + timedelta(hours=10)
            t_status = "resolved" if opened.date() < END - timedelta(days=14) or rng.random() < 0.4 \
                else rng.choice(["open", "pending"])
            closed = (opened + timedelta(hours=rng.randint(2, 240))).isoformat(sep=" ") \
                if t_status == "resolved" else None
            tickets.append([cid, oid, opened.isoformat(sep=" "), closed, _pick(rng, PRIORITIES), t_status, topic])

    # General inquiries not tied to an order.
    for _ in range(120):
        c = rng.choice(customers)
        opened = datetime.combine(_day(rng, START, END), datetime.min.time()) + timedelta(hours=rng.randint(8, 18))
        t_status = "resolved" if opened.date() < END - timedelta(days=14) else rng.choice(["open", "pending"])
        closed = (opened + timedelta(hours=rng.randint(1, 72))).isoformat(sep=" ") if t_status == "resolved" else None
        tickets.append([c[0], None, opened.isoformat(sep=" "), closed, _pick(rng, PRIORITIES), t_status,
                        rng.choice(["billing", "account"])])

    tickets.sort(key=lambda t: t[2])
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", items)
    conn.executemany("INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     [(i, *t) for i, t in enumerate(tickets, 1)])
    conn.commit()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("regions", "sales_reps", "customers", "categories", "products",
                        "orders", "order_items", "support_tickets")}
    conn.execute("VACUUM")
    conn.close()
    return counts


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    counts = build(path)
    print(f"Built {path}")
    for table, n in counts.items():
        print(f"  {table:<16} {n:>6} rows")


if __name__ == "__main__":
    main()
