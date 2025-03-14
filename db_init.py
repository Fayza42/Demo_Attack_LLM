import sqlite3
import os
import logging
from sqlite3 import Error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_database():
    """Create the SQLite database and its tables"""
    try:
        # Supprime la base si elle existe
        if os.path.exists('ecommerce.db'):
            os.remove('ecommerce.db')
            logger.info("Existing database removed")
    
        # Connexion à la base
        conn = sqlite3.connect('ecommerce.db')
        cursor = conn.cursor()
        
        # Active les clés étrangères
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Création des tables
        cursor.executescript("""
        -- Categories
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            name TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (parent_id) REFERENCES categories(id)
        );

        -- Brands
        CREATE TABLE brands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            country TEXT,
            website TEXT
        );

        -- Products
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER NOT NULL,
            brand_id INTEGER NOT NULL,
            reference TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT,
            price DECIMAL(10,2) NOT NULL CHECK (price >= 0),
            weight_grams INTEGER CHECK (weight_grams > 0),
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES categories(id),
            FOREIGN KEY (brand_id) REFERENCES brands(id)
        );

        -- Customers
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            phone TEXT,
            birth_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Customer Addresses
        CREATE TABLE customer_addresses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            address_type TEXT NOT NULL,
            street_line1 TEXT NOT NULL,
            street_line2 TEXT,
            postal_code TEXT NOT NULL,
            city TEXT NOT NULL,
            country TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        -- Orders
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled')),
            shipping_address_id INTEGER NOT NULL,
            billing_address_id INTEGER NOT NULL,
            total_amount DECIMAL(10,2) NOT NULL CHECK (total_amount >= 0),
            shipping_fee DECIMAL(10,2) NOT NULL CHECK (shipping_fee >= 0),
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (shipping_address_id) REFERENCES customer_addresses(id),
            FOREIGN KEY (billing_address_id) REFERENCES customer_addresses(id)
        );

        -- Order Items
        CREATE TABLE order_items (
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL CHECK (quantity > 0),
            unit_price DECIMAL(10,2) NOT NULL CHECK (unit_price >= 0),
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        -- Inventory
        CREATE TABLE inventory (
            product_id INTEGER PRIMARY KEY,
            quantity INTEGER NOT NULL DEFAULT 0 CHECK (quantity >= 0),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        -- Create indexes for better performance
        CREATE INDEX idx_categories_parent ON categories(parent_id);
        CREATE INDEX idx_products_category ON products(category_id);
        CREATE INDEX idx_products_brand ON products(brand_id);
        CREATE INDEX idx_customer_addresses_customer ON customer_addresses(customer_id);
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_addresses ON orders(shipping_address_id, billing_address_id);
        CREATE INDEX idx_order_items_order ON order_items(order_id);
        CREATE INDEX idx_order_items_product ON order_items(product_id);
        CREATE INDEX idx_inventory_update ON inventory(last_updated);
        """)
        
        conn.commit()
        logger.info("Database created successfully")
        
    except Error as e:
        logger.error(f"Error creating database: {e}")
        if conn:
            conn.rollback()
        raise
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    try:
        create_database()
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        exit(1)