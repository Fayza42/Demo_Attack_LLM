import sqlite3
import random
import logging
from sqlite3 import Error
from data_generator import generate_all_data
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Context manager pour la connexion à la base de données"""
    conn = None
    try:
        conn = sqlite3.connect('ecommerce.db')
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    except Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def validate_data(data):
    """Valide les données avant insertion"""
    try:
        # Vérification basique des données
        assert len(data['categories']) > 0, "No categories data"
        assert len(data['brands']) > 0, "No brands data"
        assert len(data['products']) > 0, "No products data"
        assert len(data['customers']) > 0, "No customers data"
        assert len(data['addresses']) > 0, "No addresses data"
        
        # Vérification des relations
        product_categories = {cat[0] for cat in data['categories']}
        product_brands = {brand[0] for brand in data['brands']}
        
        for product in data['products']:
            assert product[1] in product_categories, f"Invalid category_id for product {product[0]}"
            assert product[2] in product_brands, f"Invalid brand_id for product {product[0]}"
            
        return True
    except AssertionError as e:
        logger.error(f"Data validation failed: {e}")
        return False

def insert_data():
    """Insère les données dans la base"""
    try:
        # Génération des données
        logger.info("Generating data...")
        data = generate_all_data()
        
        # Validation des données
        if not validate_data(data):
            logger.error("Data validation failed")
            return False
            
        # Insertion des données
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            logger.info("Inserting categories...")
            cursor.executemany("""
                INSERT INTO categories (id, parent_id, name, description)
                VALUES (?, ?, ?, ?)
            """, data['categories'])
            
            logger.info("Inserting brands...")
            cursor.executemany("""
                INSERT INTO brands (id, name, country, website)
                VALUES (?, ?, ?, ?)
            """, data['brands'])
            
            logger.info("Inserting products...")
            cursor.executemany("""
                INSERT INTO products (id, category_id, brand_id, reference, name, 
                                    description, price, weight_grams, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data['products'])
            
            logger.info("Inserting customers...")
            cursor.executemany("""
                INSERT INTO customers (id, email, password_hash, first_name, last_name,
                                     phone, birth_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, data['customers'])
            
            logger.info("Inserting addresses...")
            cursor.executemany("""
                INSERT INTO customer_addresses (id, customer_id, address_type, street_line1,
                                             street_line2, postal_code, city, country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, data['addresses'])
            
            logger.info("Initializing inventory...")
            cursor.executemany("""
                INSERT INTO inventory (product_id, quantity)
                VALUES (?, ?)
            """, [(product[0], random.randint(0, 100)) for product in data['products']])

            logger.info("Inserting orders...")
            cursor.executemany("""
                INSERT INTO orders (id, customer_id, order_date, status,
                                  shipping_address_id, billing_address_id,
                                  total_amount, shipping_fee)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, data['orders'])

            logger.info("Inserting order items...")
            cursor.executemany("""
                INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                VALUES (?, ?, ?, ?)
            """, data['order_items'])
            
            conn.commit()
            logger.info("Data insertion completed successfully")
            return True
            
    except Error as e:
        logger.error(f"Error inserting data: {e}")
        return False

if __name__ == "__main__":
    insert_data()