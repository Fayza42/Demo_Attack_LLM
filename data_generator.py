from faker import Faker
import random
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker('fr_FR')

# Configuration
CONFIG = {
    'NB_PRODUCTS': 100,
    'NB_CUSTOMERS': 50,
    'MAX_ADDRESSES_PER_CUSTOMER': 3,
    'MAX_ORDERS_PER_CUSTOMER': 5,
    'MAX_ITEMS_PER_ORDER': 5
}

# Données des produits pharmaceutiques
PHARMA_PRODUCTS = {
    "Soins du visage": {
        "Crèmes": [
            {
                "name": "Crème Hydratante Tolériane",
                "description": "Crème hydratante apaisante pour peaux sensibles. Sans parfum, sans paraben. Hydratation 48h.",
                "brand": "La Roche-Posay",
                "price": 15.99,
                "weight": 40
            },
            {
                "name": "Crème Anti-rides Redermic R",
                "description": "Soin anti-âge au rétinol pur. Combat les rides installées et le relâchement cutané.",
                "brand": "La Roche-Posay",
                "price": 32.99,
                "weight": 30
            }
        ],
        "Sérums": [
            {
                "name": "Sérum Vitamine C Pure",
                "description": "Sérum anti-oxydant concentré en vitamine C pure. Éclat et protection anti-âge.",
                "brand": "Vichy",
                "price": 28.50,
                "weight": 20
            },
            {
                "name": "Hyalu B5 Sérum",
                "description": "Sérum réparateur à l'acide hyaluronique. Hydrate et répare la barrière cutanée.",
                "brand": "La Roche-Posay",
                "price": 39.99,
                "weight": 30
            }
        ],
        "Nettoyants": [
            {
                "name": "Eau Micellaire Sensibio H2O",
                "description": "Eau micellaire démaquillante pour peaux sensibles. Nettoie en douceur.",
                "brand": "Bioderma",
                "price": 12.99,
                "weight": 250
            },
            {
                "name": "Gel Moussant Effaclar",
                "description": "Gel nettoyant purifiant pour peaux grasses à tendance acnéique.",
                "brand": "La Roche-Posay",
                "price": 15.50,
                "weight": 200
            }
        ]
    },
    "Soins du corps": {
        "Crèmes corps": [
            {
                "name": "Lipikar Baume AP+M",
                "description": "Baume relipidant anti-grattage pour peaux très sèches à tendance atopique.",
                "brand": "La Roche-Posay",
                "price": 19.99,
                "weight": 400
            },
            {
                "name": "Huile Lavante Relipidante",
                "description": "Huile lavante pour peaux sèches à très sèches. Nettoie et nourrit.",
                "brand": "Eucerin",
                "price": 16.99,
                "weight": 400
            }
        ]
    },
    "Compléments alimentaires": {
        "Vitamines": [
            {
                "name": "Vitamine D3 1000 UI",
                "description": "Contribue au maintien d'une ossature normale et au fonctionnement normal du système immunitaire.",
                "brand": "Bioderma",
                "price": 9.99,
                "weight": 30
            },
            {
                "name": "Magnésium B6",
                "description": "Réduit la fatigue et contribue à un métabolisme énergétique normal.",
                "brand": "Vichy",
                "price": 12.99,
                "weight": 60
            }
        ],
        "Minéraux": [
            {
                "name": "Fer + Vitamine C",
                "description": "Contribue à réduire la fatigue et au transport normal de l'oxygène dans l'organisme.",
                "brand": "Bioderma",
                "price": 14.99,
                "weight": 45
            }
        ]
    },
    "Hygiène": {
        "Dentaire": [
            {
                "name": "Dentifrice Sensibilité",
                "description": "Soulage les dents sensibles. Protection longue durée contre la sensibilité dentaire.",
                "brand": "Neutrogena",
                "price": 7.99,
                "weight": 75
            },
            {
                "name": "Bain de Bouche Quotidien",
                "description": "Protection contre la plaque dentaire et maintien d'une bonne hygiène bucco-dentaire.",
                "brand": "SVR",
                "price": 6.99,
                "weight": 500
            }
        ]
    },
    "Matériel médical": {
        "Pansements": [
            {
                "name": "Pansements Stériles Waterproof",
                "description": "Pansements imperméables stériles. Protection optimale contre l'eau et les bactéries.",
                "brand": "Neutrogena",
                "price": 5.99,
                "weight": 20
            }
        ],
        "Thermomètres": [
            {
                "name": "Thermomètre Digital Flex",
                "description": "Thermomètre digital flexible avec embout souple. Mesure précise en 10 secondes.",
                "brand": "Bioderma",
                "price": 12.99,
                "weight": 50
            }
        ]
    }
}

def generate_categories():
    """Génère la hiérarchie des catégories"""
    categories_data = []
    category_id = 1

    main_categories = [
        ("Soins du visage", "Produits pour le soin du visage"),
        ("Soins du corps", "Produits pour le soin du corps"),
        ("Hygiène", "Produits d'hygiène quotidienne"),
        ("Compléments alimentaires", "Vitamines et suppléments"),
        ("Matériel médical", "Équipement médical de base")
    ]

    sub_categories = {
        "Soins du visage": ["Crèmes", "Sérums", "Masques", "Nettoyants"],
        "Soins du corps": ["Crèmes corps", "Huiles", "Gels douche", "Déodorants"],
        "Hygiène": ["Dentaire", "Capillaire", "Intime", "Mains"],
        "Compléments alimentaires": ["Vitamines", "Minéraux", "Probiotiques", "Oméga 3"],
        "Matériel médical": ["Pansements", "Thermomètres", "Masques", "Désinfectants"]
    }

    main_category_ids = {}

    # Catégories principales
    for name, desc in main_categories:
        categories_data.append((category_id, None, name, desc))
        main_category_ids[name] = category_id
        category_id += 1

    # Sous-catégories
    for main_cat, sub_cats in sub_categories.items():
        parent_id = main_category_ids[main_cat]
        for sub_name in sub_cats:
            categories_data.append((category_id, parent_id, sub_name, f"Sous-catégorie de {main_cat}"))
            category_id += 1

    logger.info(f"Generated {len(categories_data)} categories")
    return categories_data

def generate_brands():
    """Génère les marques"""
    brands = [
        ("Avène", "France", "www.avene.fr"),
        ("La Roche-Posay", "France", "www.laroche-posay.fr"),
        ("Vichy", "France", "www.vichy.fr"),
        ("Bioderma", "France", "www.bioderma.fr"),
        ("Nuxe", "France", "www.nuxe.com"),
        ("Eucerin", "Allemagne", "www.eucerin.fr"),
        ("Neutrogena", "USA", "www.neutrogena.fr"),
        ("SVR", "France", "www.svr.com"),
        ("Caudalie", "France", "www.caudalie.com"),
        ("Uriage", "France", "www.uriage.com")
    ]
    brands_data = [(i, name, country, website) for i, (name, country, website) in enumerate(brands, 1)]
    logger.info(f"Generated {len(brands_data)} brands")
    return brands_data

def generate_products(nb_products=None):
    """Génère les produits"""
    if nb_products is None:
        nb_products = CONFIG['NB_PRODUCTS']

    products = []
    product_id = 1

    # Dictionnaire pour mapper les catégories et marques aux IDs
    categories = {cat[2]: cat[0] for cat in generate_categories()}
    brands = {brand[1]: brand[0] for brand in generate_brands()}

    # Génération des produits prédéfinis
    for main_category, subcategories in PHARMA_PRODUCTS.items():
        for subcategory, products_list in subcategories.items():
            for product_data in products_list:
                if subcategory in categories:
                    products.append((
                        product_id,
                        categories[subcategory],
                        brands[product_data["brand"]],
                        f"REF{str(product_id).zfill(6)}",
                        product_data["name"],
                        product_data["description"],
                        product_data["price"],
                        product_data["weight"],
                        1,  # active
                        datetime.now()
                    ))
                    product_id += 1

    logger.info(f"Generated {len(products)} products")
    return products

# Le reste du code reste identique
def generate_customers(nb_customers=None):
    """Génère les clients"""
    if nb_customers is None:
        nb_customers = CONFIG['NB_CUSTOMERS']
        
    customers = []
    for i in range(1, nb_customers + 1):
        email = fake.email()
        password_hash = "motdepasse123"  # Non sécurisé intentionnellement
        first_name = fake.first_name()
        last_name = fake.last_name()
        phone = fake.phone_number()
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90)
        created_at = fake.date_time_between(start_date='-1y')
        
        customers.append((i, email, password_hash, first_name, last_name,
                         phone, birth_date, created_at))
    
    logger.info(f"Generated {len(customers)} customers")
    return customers

def generate_customer_addresses(customers_data):
    """Génère les adresses des clients"""
    addresses = []
    address_id = 1
    
    for customer_id, *_ in customers_data:
        nb_addresses = random.randint(1, CONFIG['MAX_ADDRESSES_PER_CUSTOMER'])
        for _ in range(nb_addresses):
            address_type = random.choice(['domicile', 'travail', 'autre'])
            street_line1 = fake.street_address()
            street_line2 = f"Apt {random.randint(1, 999)}" if random.random() > 0.7 else None
            postal_code = fake.postcode()
            city = fake.city()
            country = "France"
            
            addresses.append((address_id, customer_id, address_type, street_line1,
                            street_line2, postal_code, city, country))
            address_id += 1
    
    logger.info(f"Generated {len(addresses)} addresses")
    return addresses

def generate_orders_and_items(customers_data, addresses_data, products_data):
    """Génère les commandes et leurs items"""
    orders = []
    order_items = []
    order_id = 1
    
    # Grouper les adresses par client
    customer_addresses = {}
    for addr in addresses_data:
        if addr[1] not in customer_addresses:  # addr[1] est customer_id
            customer_addresses[addr[1]] = []
        customer_addresses[addr[1]].append(addr[0])  # addr[0] est address_id
    
    for customer_id, *_ in customers_data:
        if customer_id not in customer_addresses:
            continue
            
        nb_orders = random.randint(0, CONFIG['MAX_ORDERS_PER_CUSTOMER'])
        customer_addrs = customer_addresses[customer_id]
        
        for _ in range(nb_orders):
            # Addresses
            shipping_address = random.choice(customer_addrs)
            billing_address = shipping_address if random.random() < 0.8 else random.choice(customer_addrs)
            
            # Date et statut
            order_date = fake.date_time_between(start_date='-1y')
            days_since_order = (datetime.now() - order_date).days
            
            if days_since_order > 30:
                status = 'delivered'
            elif days_since_order > 14:
                status = random.choice(['delivered', 'shipped'])
            elif days_since_order > 7:
                status = random.choice(['shipped', 'confirmed'])
            else:
                status = random.choice(['pending', 'confirmed'])
            
            # Items
            nb_items = random.randint(1, CONFIG['MAX_ITEMS_PER_ORDER'])
            order_total = 0
            shipping_fee = random.choice([0, 4.99, 7.99])
            
            selected_products = random.sample(products_data, nb_items)
            for product in selected_products:
                product_id = product[0]
                unit_price = product[6]  # price is at index 6
                quantity = random.randint(1, 3)
                
                order_items.append((order_id, product_id, quantity, unit_price))
                order_total += unit_price * quantity
            
            order_total += shipping_fee
            
            orders.append((order_id, customer_id, order_date, status,
                          shipping_address, billing_address, order_total, shipping_fee))
            order_id += 1
    
    logger.info(f"Generated {len(orders)} orders with {len(order_items)} items")
    return orders, order_items

def generate_all_data():
    """Génère toutes les données"""
    categories = generate_categories()
    brands = generate_brands()
    products = generate_products()
    customers = generate_customers()
    addresses = generate_customer_addresses(customers)
    orders, order_items = generate_orders_and_items(customers, addresses, products)
    
    return {
        'categories': categories,
        'brands': brands,
        'products': products,
        'customers': customers,
        'addresses': addresses,
        'orders': orders,
        'order_items': order_items
    }

if __name__ == "__main__":
    data = generate_all_data()
    print("Données générées avec succès!")