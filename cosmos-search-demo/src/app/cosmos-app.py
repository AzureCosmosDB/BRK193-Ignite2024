import openai
import streamlit as st
from openai import AzureOpenAI
import os
import numpy as np
import json
from datetime import datetime
import pandas as pd
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Ignite 2024 Demo", layout="wide", initial_sidebar_state="expanded")
# UI text strings
page_title = "Azure Cosmos DB - Search Demo"
page_helper = "Showcasing DiskANN (GA), full text search, text ranking, and hybrid search released at Ignite 2024."
empty_search_helper = "Enter text relating to an area of research to get started."
semantic_search_header = "Search input"
semantic_search_placeholder = "A Cantorian fractal spacetime"
vector_search_label = "Similarity search"
full_text_ranking_label = "Full text ranking"
full_text_search_label = "Full text search"
venue_list_header = "Research papers"
hybrid_search_label = "Hybrid search"

# Initialize global variables for Cosmos DB client, database, and containers
if "cosmos_client" not in st.session_state:
    endpoint = os.getenv("AZURE_COSMOSDB_ENDPOINT")
    key = os.getenv("AZURE_COSMOSDB_KEY")
    st.session_state.cosmos_client = CosmosClient(endpoint, credential=key)
    database_name = 'ignite2024demo'  # Replace with your database name
    st.session_state.cosmos_database = st.session_state.cosmos_client.create_database_if_not_exists(database_name)

    # Define the vector property and dimensions
    cosmos_vector_property = "embedding"
    cosmos_full_text_property = "text"
    openai_embeddings_dimensions = 1536

    # policies and indexes
    full_text_policy = {
        "defaultLanguage": "en-US",
        "fullTextPaths": [
            {
                "path": "/" + cosmos_full_text_property,
                "language": "en-US",
            }
        ]
    }
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/" + cosmos_vector_property,
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": openai_embeddings_dimensions
            },
        ]
    }
    qflat_indexing_policy = {
        "includedPaths": [
            {"path": "/*"}
        ],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"}
        ],
        "vectorIndexes": [
            {
                "path": "/" + cosmos_vector_property,
                "type": "quantizedFlat",
            }
        ],
        "fullTextIndexes": [
            {
                "path": "/" + cosmos_full_text_property
            }
        ]
    }

    diskann_indexing_policy = {
        "includedPaths": [
            {"path": "/*"}
        ],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"}
        ],
        "vectorIndexes": [
            {
                "path": "/" + cosmos_vector_property,
                "type": "diskANN",
            }
        ],
        "fullTextIndexes": [
            {
                "path": "/" + cosmos_full_text_property
            }
        ]
    }

    # Create listings_search container without any index
    container_name = 'search'
    st.session_state.cosmos_container = st.session_state.cosmos_database.create_container_if_not_exists(
        id=container_name,
        partition_key=PartitionKey(path="/id"),
        full_text_policy=full_text_policy,
        vector_embedding_policy=vector_embedding_policy,
        offer_throughput=10000
    )


    # Create listings_search_qflat container with QFLAT vector index
    container_name_qflat = 'search_qflat'
    st.session_state.cosmos_container_qflat = st.session_state.cosmos_database.create_container_if_not_exists(
        id=container_name_qflat,
        partition_key=PartitionKey(path="/id"),
        full_text_policy=full_text_policy,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=qflat_indexing_policy,
        offer_throughput=10000
    )

    # Create listings_search_diskann container with DiskANN vector index
    container_name_diskann = 'search_diskann'
    st.session_state.cosmos_container_diskann = st.session_state.cosmos_database.create_container_if_not_exists(
        id=container_name_diskann,
        partition_key=PartitionKey(path="/id"),
        full_text_policy=full_text_policy,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=diskann_indexing_policy,
        offer_throughput=10000
    )

# Initialize session state variables
if "embedding_gen_time" not in st.session_state:
    st.session_state.embedding_gen_time = ""
if "query_time" not in st.session_state:
    st.session_state.query_time = ""
if "ru_consumed" not in st.session_state:
    st.session_state.ru_consumed = ""
if "executed_query" not in st.session_state:
    st.session_state.executed_query = ""
if "server_query_time" not in st.session_state:
    st.session_state.server_query_time = ""

# Function to log times
def log_time(start):
    end = time.perf_counter()
    elapsed_time = end - start
    return f"{elapsed_time:.4f} seconds"

# Initialize the embedding client only once
if "embedding_client" not in st.session_state:
    st.session_state.embedding_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_APIKEY"),
        api_version="2023-05-15",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Handler functions
def embedding_query(text_input):
    print("text_input", text_input)
    start_time = time.perf_counter()
    response = st.session_state.embedding_client.embeddings.create(
        input=text_input,
        model="text-embedding-ada-002"  # Use the appropriate model
    )

    json_response = response.model_dump_json(indent=2)
    parsed_response = json.loads(json_response)
    embedding = parsed_response['data'][0]['embedding']
    st.session_state.embedding_gen_time = log_time(start_time)
    print(f"Embedding generation time: {st.session_state.embedding_gen_time}")
    return embedding

def handler_vector_search(indices, ask):
    emb = embedding_query(ask)
    num_results = 10

    # Query strings
    vector_search_query = f'''
    SELECT TOP {num_results} l.id, l.title, l.text, VectorDistance(l.embedding, {emb}) as SimilarityScore
    FROM l
    ORDER BY VectorDistance(l.embedding,{emb})
    '''

    obfuscated_query = vector_search_query.replace(str(emb), "REDACTED")

    container = {
        'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = obfuscated_query
        results = container.query_items(vector_search_query, enable_cross_partition_query=True, populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        st.session_state.suggested_listings = pd.DataFrame(results_list)
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred: {e}")

def handler_text_search(indices, text, search_type):
    num_results = 10

    # Tokenize text into individual words
    keywords = text.split()  # Split the text into words
    formatted_keywords = ', '.join(f'"{keyword}"' for keyword in keywords)
    print(formatted_keywords)# Format keywords for query

    # Construct the query string with tokenized keywords
    if search_type == "all keywords":
        full_text_search_query = f'''
        SELECT TOP {num_results} l.id, l.title, l.text
        FROM l
        WHERE FullTextContainsAll(l.text, {formatted_keywords})
        '''
    else:
        full_text_search_query = f'''
        SELECT TOP {num_results} l.id, l.title, l.text
        FROM l
        WHERE FullTextContainsAny(l.text, {formatted_keywords})
        '''

    container = {
        'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = full_text_search_query
        results = container.query_items(full_text_search_query, enable_cross_partition_query=True, populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        st.session_state.suggested_listings = pd.DataFrame(results_list)
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred: {e}")

def handler_text_ranking(indices, text):
    num_results = 10

    # Tokenize text into individual words
    keywords = text.split()  # Split the text into words
    formatted_keywords = ', '.join(f'"{keyword}"' for keyword in keywords)  # Format keywords for query

    # Construct the query string with tokenized keywords

    full_text_ranking_query = f'''
    SELECT TOP {num_results} l.id, l.title, l.text
    FROM l
    ORDER BY RANK FullTextScore(l.text, [{formatted_keywords}])
    '''

    container = {
        'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = full_text_ranking_query
        results = container.query_items(full_text_ranking_query, enable_cross_partition_query=True,populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        st.session_state.suggested_listings = pd.DataFrame(results_list)
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred: {e}")

def handler_hybrid_ranking(indices, text):
    num_results = 10
    emb = embedding_query(text)
    # Tokenize text into individual words
    keywords = text.split()  # Split the text into words
    formatted_keywords = ', '.join(f'"{keyword}"' for keyword in keywords)  # Format keywords for query

    # Construct the query string with tokenized keywords

    full_hybrid_ranking_query = f'''
    SELECT TOP {num_results} l.id, l.title, l.text
    FROM l
    ORDER BY RANK RRF(FullTextScore(l.text,[{formatted_keywords}]),VectorDistance(l.embedding, {emb}))
    '''

    obfuscated_query = full_hybrid_ranking_query.replace(str(emb), "REDACTED")

    container = {
        'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = obfuscated_query
        results = container.query_items(full_hybrid_ranking_query, enable_cross_partition_query=True,populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        st.session_state.suggested_listings = pd.DataFrame(results_list)
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred: {e}")

# UI elements
def render_cta_link(url, label, font_awesome_icon):
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">',
        unsafe_allow_html=True)
    button_code = f'''<a href="{url}" target=_blank><i class="fa {font_awesome_icon}"></i> {label}</a>'''
    return st.markdown(button_code, unsafe_allow_html=True)

def parse_server_query_time(query_metrics):
    metrics_parts = query_metrics.split(";")
    total_execution_time_ms = next(
        (part.split("=")[1] for part in metrics_parts if "totalExecutionTimeInMs" in part), "0"
    )
    total_execution_time_s = float(total_execution_time_ms) / 1000
    return f"{total_execution_time_s:.4f} seconds"

def render_search():
    search_disabled = True
    with st.sidebar:
        st.selectbox(label="Index", options=['No Index', 'QFLAT & Full Text Search Index', 'DiskANN & Full Text Search Index'], index=0, key="index_selection")
        st.text_input(label=semantic_search_header, placeholder=semantic_search_placeholder, key="user_query")

        if "user_query" in st.session_state and st.session_state.user_query != "":
            search_disabled = False

        st.button(label=vector_search_label, key="location_search", disabled=search_disabled,
                  on_click=handler_vector_search, args=(st.session_state.index_selection, st.session_state.user_query))

        # Button for Full Text Ranking search using handler_text_ranking
        st.button(label=full_text_ranking_label, key="full_text_ranking", disabled=search_disabled,
                  on_click=handler_text_ranking, args=(st.session_state.index_selection, st.session_state.user_query))

        # Button for Hybrid Ranking search using handler_hybrid_ranking
        st.button(label=hybrid_search_label, key="hybrid_search", disabled=search_disabled,
                  on_click=handler_hybrid_ranking, args=(st.session_state.index_selection, st.session_state.user_query))

        search_type = st.radio("Search type", options=["all keywords", "any keywords"], key="full_text_search_type")

        st.button(label=full_text_search_label, key="full_text_search", disabled=search_disabled,
                  on_click=handler_text_search, args=(st.session_state.index_selection, st.session_state.user_query, search_type))



        st.write("---")
        render_cta_link(url="https://azurecosmosdb.github.io/gallery/", label="Cosmos DB Samples Gallery", font_awesome_icon="fa-cosmosdb")
        render_cta_link(url="https://github.com/AzureCosmosDB/BRK193-Ignite2024/cosmos-search-demo", label="GitHub", font_awesome_icon="fa-github")

def render_search_result():
    col1 = st.container()
    col1.write(f"Executed query: {st.session_state.executed_query}")
    col1.write(f"Embedding generation time: {st.session_state.embedding_gen_time}")
    col1.write(f"Total end-to-end query execution time: {st.session_state.query_time}")
    col1.write(f"Total server query execution time: {st.session_state.server_query_time}")
    col1.write(f"RU consumed: {st.session_state.ru_consumed}")
    col1.write(f"Found {len(st.session_state.suggested_listings)} records.")
    col1.table(st.session_state.suggested_listings)

# Main execution
render_search()

st.title(page_title)
st.write(page_helper)
st.write("---")

if "suggested_listings" not in st.session_state:
    st.write(empty_search_helper)
else:
    render_search_result()