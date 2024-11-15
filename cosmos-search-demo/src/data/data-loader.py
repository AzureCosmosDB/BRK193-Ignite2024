import argparse
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from asyncio import Semaphore

import requests
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Initialize Cosmos DB client
endpoint = os.getenv("AZURE_COSMOSDB_ENDPOINT")
key = os.getenv("AZURE_COSMOSDB_KEY")

# Initialize OpenAI client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_APIKEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def initialize_cosmos(database_name):
    client = CosmosClient(endpoint, key)
    database = client.get_database_client(database_name)
    container_names = ['search', 'search_qflat', 'search_diskann']
    containers = {name: database.get_container_client(name) for name in container_names}
    return containers


def load_json_data(file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Handle URLs
        response = requests.get(file_path)
        response.raise_for_status()  # Raise an error if the request fails
        return response.json()
    elif os.path.exists(file_path):
        # Handle local file paths
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        raise ValueError(f"Invalid file path or URL: {file_path}")


def generate_embedding(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
    json_response = response.model_dump_json(indent=2)
    parsed_response = json.loads(json_response)
    return parsed_response['data'][0]['embedding']


def upsert_item_sync(container, item):
    try:
        container.upsert_item(body=item)
    except exceptions.CosmosHttpResponseError as e:
        print(f"Failed to insert document: {e.message}")


async def upsert_items_async(containers, items, text_field_name, max_concurrency, vector_field_name=None, re_embed=False):
    semaphore = Semaphore(max_concurrency)
    loop = asyncio.get_event_loop()
    progress_counter = 0

    async def process_item(item):
        nonlocal progress_counter
        async with semaphore:
            # Rename vector_field_name to embedding if present
            if vector_field_name in item:
                # Re-embed the text if re_embed is True
                if re_embed:
                    item['embedding'] = generate_embedding(item[text_field_name])
                    # get rid of previous vector_field_name, not needed anymore
                    item.pop(vector_field_name)
                    # Rename text_field_name to text so that it matches the streamlit app
                    item['text'] = item.pop(text_field_name)
                else:
                    # Rename vector_field_name to embedding so that it matches the streamlit app
                    item['embedding'] = item.pop(vector_field_name)
                    # Rename text_field_name to text so that it matches the streamlit app
                    item['text'] = item.pop(text_field_name)
            # Generate embedding for text_field_name if present
            elif text_field_name in item:
                item['embedding'] = generate_embedding(item[text_field_name])
                # Rename text_field_name to text so that it matches the streamlit app
                item['text'] = item.pop(text_field_name)

            # Upsert item to all containers
            for container in containers.values():
                await loop.run_in_executor(None, upsert_item_sync, container, item)

            # Update progress counter
            progress_counter += 1
            if progress_counter % 100 == 0:
                print(f"{progress_counter} records processed.")

    # Create tasks for processing items
    tasks = [process_item(item) for item in items]
    await asyncio.gather(*tasks)



async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Upsert items into Cosmos DB.")
    parser.add_argument("--text_field_name", required=True, help="The name of the field containing text to generate embeddings.")
    parser.add_argument("--path_to_json_array", required=True, help="The path to the JSON file containing the array of items.")
    parser.add_argument("--database_name", required=True, help="The name of the Cosmos DB database.")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum number of concurrent upsert operations.")
    parser.add_argument("--vector_field_name", help="The name of the field containing pre-generated embeddings.")
    parser.add_argument("--re_embed", type=bool, default=False, help="Whether to re-embed the text or not.")
    args = parser.parse_args()

    # Initialize containers and load data
    containers = initialize_cosmos(args.database_name)
    items = load_json_data(args.path_to_json_array)

    # Call the upsert function with the specified parameters
    await upsert_items_async(containers, items, text_field_name=args.text_field_name, max_concurrency=args.concurrency, vector_field_name=args.vector_field_name, re_embed=args.re_embed)

    # how to call this function
    # python src/data/data-loader.py --text_field_name "overview" --path_to_json_array "https://raw.githubusercontent.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples/refs/heads/main/DataSet/Movies/MovieLens-4489-256D.json" --database_name "ignite2024demo" --concurrency 20 --vector_field_name "vector" --re_embed True


if __name__ == "__main__":
    asyncio.run(main())
