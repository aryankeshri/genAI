import os
import random
import sqlite3


if os.path.exists("atliq_tshirts.db"):
    os.remove("atliq_tshirts.db")

con = sqlite3.connect("atliq_tshirts.db")

cur = con.cursor()


def create_tables():
    query = """
        CREATE TABLE t_shirts (
            t_shirt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT CHECK(brand IN ('Van Huesen','Levi','Adidas')) NOT NULL,
            color TEXT CHECK(color IN ('Red', 'Blue', 'Black', 'White')) NOT NULL,
            size TEXT CHECK(size IN ('XS', 'S', 'M', 'L', 'XL')) NOT NULL,
            price INTEGER CHECK (price BETWEEN 10 AND 50),
            stock_quantity INTEGER NOT NULL,
            UNIQUE(brand, color, size) ON CONFLICT REPLACE
        );
    """
    cur.execute(query)

    query = """
        CREATE TABLE discounts (
            discount_id INTEGER PRIMARY KEY AUTOINCREMENT,
            t_shirt_id INTEGER NOT NULL,
            pct_discount DECIMAL(5,2) CHECK (pct_discount BETWEEN 0 AND 100),
            FOREIGN KEY(t_shirt_id) REFERENCES t_shirts(t_shirt_id)
        );
    """
    cur.execute(query)


def insert_table_data():
    values = ""

    for i in range(0, 100):
        brand = ('Van Huesen', 'Levi', 'Adidas')[random.randint(0, 2)]
        color = ('Red', 'Blue', 'Black', 'White')[random.randint(0, 3)]
        size = ('XS', 'S', 'M', 'L', 'XL')[random.randint(0, 4)]
        values += f"""('{brand}','{color}','{size}',{random.randint(10, 50)},{random.randint(10, 50)}),"""

    query = f"""
    INSERT INTO t_shirts (brand, color, size, price, stock_quantity)
    VALUES {values.rstrip(',')};
    """
    print(query)
    cur.execute(query)
    con.commit()

    query = """
    INSERT INTO discounts (t_shirt_id, pct_discount)
    VALUES
    (1, 10.00),
    (2, 15.00),
    (3, 20.00),
    (4, 5.00),
    (5, 25.00),
    (6, 10.00),
    (7, 30.00),
    (8, 35.00),
    (9, 40.00),
    (10, 45.00);
    """
    cur.execute(query)
    con.commit()


def database_creater():
    try:
        create_tables()
        insert_table_data()
    except sqlite3.Error as error:
        print("Failed to connect with sqlite3 database", error)
    finally:
        cur.close()
        con.close()


if __name__ == '__main__':
    database_creater()
