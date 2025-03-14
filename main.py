import os
import logging
import sys
import argparse
import sqlite3
from db_init import create_database
from data_inserter import insert_data
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'db_creation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Vérifie que toutes les conditions sont réunies pour exécuter le script"""
    try:
        import faker
        import sqlite3
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please install required packages using: pip install -r requirements.txt")
        return False
    
    # Vérifier les permissions du répertoire
    try:
        if os.path.exists('ecommerce.db'):
            os.access('ecommerce.db', os.W_OK)
        else:
            # Vérifier qu'on peut écrire dans le répertoire
            with open('test_write_permission', 'w') as f:
                f.write('test')
            os.remove('test_write_permission')
    except (IOError, OSError) as e:
        logger.error(f"Directory permission error: {e}")
        return False
        
    return True

def verify_database():
    """Vérifie l'intégrité de la base de données créée"""
    try:
        conn = sqlite3.connect('ecommerce.db')
        cursor = conn.cursor()
        
        # Vérifier que toutes les tables existent
        tables = [
            'categories', 'brands', 'products', 'customers',
            'customer_addresses', 'orders', 'order_items', 'inventory'
        ]
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"Table {table}: {count} records")
            
        # Vérifier les clés étrangères
        cursor.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()
        if fk_violations:
            logger.error(f"Foreign key violations found: {fk_violations}")
            return False
            
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database verification failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Create and populate e-commerce database')
    parser.add_argument('--force', action='store_true', 
                      help='Force database recreation if it exists')
    parser.add_argument('--no-data', action='store_true',
                      help='Create database structure only, without test data')
    parser.add_argument('--verify', action='store_true',
                      help='Verify database after creation')
    return parser.parse_args()

def main():
    """Point d'entrée principal du script"""
    logger.info("Starting e-commerce database creation...")
    
    # Parse les arguments
    args = parse_arguments()
    
    # Vérification des prérequis
    if not check_prerequisites():
        logger.error("Prerequisites check failed")
        sys.exit(1)
    
    # Vérifier si la base existe déjà
    if os.path.exists('ecommerce.db') and not args.force:
        logger.error("Database already exists. Use --force to recreate it.")
        sys.exit(1)
    
    try:
        # 1. Création de la base de données
        logger.info("Creating database structure...")
        create_database()
        
        # 2. Insertion des données si demandé
        if not args.no_data:
            logger.info("Inserting test data...")
            if not insert_data():
                raise Exception("Data insertion failed")
        
        # 3. Vérification si demandée
        if args.verify:
            logger.info("Verifying database...")
            if not verify_database():
                raise Exception("Database verification failed")
        
        # 4. Afficher les statistiques finales
        db_size = os.path.getsize('ecommerce.db') / (1024 * 1024)  # Taille en MB
        logger.info("Database creation completed successfully!")
        logger.info(f"Database file: {os.path.abspath('ecommerce.db')}")  # Correction de la parenthèse manquante
        logger.info(f"Database size: {db_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
        
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()