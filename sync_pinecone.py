import os
import time
import mysql.connector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone,ServerlessSpec
from tqdm.auto import tqdm
from dotenv import load_dotenv
import pandas as pd



load_dotenv()


# Pinecone Configuration
api_key=os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=api_key)

spec=ServerlessSpec(
    cloud="aws",region="us-east-1"
)

index_name="shop-product-catalog"
existing_index=[ index_info['name'] for index_info in pc.list_indexes()]

# check if index is already exist
if index_name not in existing_index:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to the index
index=pc.Index(index_name)
time.sleep(1)

# connect to MySQL
db_connection=mysql.connector.connect(
    host='localhost',
    user='root',
    password=os.getenv('DB_PASSWORD'),
    database='shopassistants'
)

cursor=db_connection.cursor()

# Google GenAI API
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def fetch_data():
    query="SELECT * FROM products"
    cursor.execute(query)
    columns=[  desc[0] for desc in cursor.description]
    data=pd.DataFrame(cursor.fetchall(),columns=columns)
    return data




def sync_with_pinecone(data):
    batch_size=100
    total_batches=(len(data)+batch_size -1) // batch_size


    for i in tqdm(range(0,len(data),batch_size),desc="Processing Batches",unit='batch',total=total_batches):
        i_end=min(len(data),i + batch_size)
        batch=data.iloc[i:i_end]

        # unique id
        ids=[str(row['ProductID']) for _,row in batch.iterrows()]

        # combine text field for embedding
        texts=[
            f"{row['Description']} {row['ProductName']} {row['ProductBrand']} {row['Gender']} {row['Price']} {row['PrimaryColor']}"
            for _,row in batch.iterrows()
        ]

        # embed texts
        embeds=embed_model.embed_documents(texts)

        # get metadata
        metadata=[
            {
                'ProductName':row['ProductName'],
                'ProductBrand':row['ProductBrand'],
                'Gender':row['Gender'],
                'Price':row['Price'],
                'PrimaryColor':row['PrimaryColor'],
                'Description':row['Description'],
            }
            for _,row in batch.iterrows()
        ]

        # upsert vectors
        with tqdm(total=len(ids),desc="Upserting Vectors",unit='vector') as upsert_vector:
            index.upsert(vectors=zip(ids,embeds,metadata))
            upsert_vector.update(len(ids))



def main():
    data=fetch_data()
    sync_with_pinecone(data)

if __name__=="__main__":
    main()



cursor.close()
db_connection.close()

