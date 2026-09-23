CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    article TEXT NOT NULL,
    client_id TEXT NOT NULL,
    qty NUMERIC(12, 3) NOT NULL,
    price NUMERIC(12, 3) NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    id SERIAL PRIMARY KEY,
    article TEXT NOT NULL,
    quantity_on_hand NUMERIC(12, 3) NOT NULL,
    warehouse TEXT NOT NULL DEFAULT 'central'
);

CREATE TABLE IF NOT EXISTS inbound (
    id SERIAL PRIMARY KEY,
    article TEXT NOT NULL,
    expected_qty NUMERIC(12, 3) NOT NULL,
    eta DATE
);

CREATE TABLE IF NOT EXISTS stockouts (
    id SERIAL PRIMARY KEY,
    article TEXT NOT NULL,
    stockout_days INTEGER NOT NULL,
    lost_sales_units NUMERIC(12, 3) NOT NULL
);

CREATE TABLE IF NOT EXISTS suppliers (
    id SERIAL PRIMARY KEY,
    article TEXT NOT NULL,
    supplier TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL,
    min_order_qty NUMERIC(12, 3) NOT NULL DEFAULT 0
);
