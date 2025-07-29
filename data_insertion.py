# import library
import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# read csv file:
csv_file='shop-product-catalog.csv'
data=pd.read_csv(csv_file)

# connect to MySQL
db_connection=mysql.connector.connect(
    host='localhost',
    user='root',
    password=os.getenv('DB_PASSWORD'),
    database='shopassistants'
)

cursor=db_connection.cursor()


for index,row in data.iterrows():
    sql="""
    INSERT INTO products (ProductID,ProductName,ProductBrand,Gender,Price,Description,PrimaryColor)
    VALUES (%s, %s,%s,%s,%s,%s,%s)
    """
    cursor.execute(sql,tuple(row))

db_connection.commit()


cursor.close()
db_connection.close()



