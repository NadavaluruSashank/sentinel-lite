CREATE OR REPLACE VIEW delivered_orders AS
WITH items AS (
    SELECT order_id,
           COUNT(*)                     AS n_items,
           COUNT(DISTINCT seller_id)    AS n_sellers,
           SUM(price)                   AS total_price,
           SUM(freight_value)           AS total_freight,
           MIN(seller_id)               AS seller_id
    FROM order_items
    GROUP BY order_id
)
SELECT o.order_id,
       c.customer_state,
       s.seller_state,
       i.n_items, i.n_sellers, i.total_price, i.total_freight,
       o.order_purchase_timestamp::timestamp        AS purchase_ts,
       o.order_estimated_delivery_date::timestamp   AS estimated_ts,
       o.order_delivered_customer_date::timestamp   AS delivered_ts,
       EXTRACT(EPOCH FROM (o.order_delivered_customer_date::timestamp
                         - o.order_purchase_timestamp::timestamp)) / 86400.0 AS delivery_days,
       EXTRACT(EPOCH FROM (o.order_estimated_delivery_date::timestamp
                         - o.order_purchase_timestamp::timestamp)) / 86400.0 AS promised_days,
       (o.order_delivered_customer_date::timestamp
          > o.order_estimated_delivery_date::timestamp)::int AS is_late
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN items i     ON i.order_id    = o.order_id
LEFT JOIN sellers s ON s.seller_id = i.seller_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL;
