
from bson import ObjectId
import gridfs
import sys
import hashlib
import os
import json

import pwd
import sys

from loguru import logger
from pymongo import MongoClient

def mongo_connection(host, username, password, authSource  ):
    """
    Establishes a connection to MongoDB using the provided credentials.

    Parameters:
    - host (str): The MongoDB server's host address.
    - username (str): The username used to authenticate the connection.
    - password (str): The password associated with the provided username.
    - auth_source (str): The authentication database.

    Returns:
    - MongoClient: A connection object to interact with the MongoDB database.

    Raises:
    - SystemExit: If unable to establish a connection to MongoDB.
    """
    
    client = MongoClient(host=host, username=username, password=password, authSource=authSource)
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        logger.error("Can not access to MongoDB...")
        logger.error("")
        logger.error(e)
        sys.exit()

    return client

def calculate_checksum(data):
    """
    Calculate the checksum value for the given data.

    Args:
        data (bytes): Data to calculate the checksum for.

    Returns:
        str: Checksum value.
    """
    checksum = hashlib.md5(data).hexdigest()
    return checksum


def store_model(client, metadata):
    """
    Stores the model using the MongoDB client and metadata.

    Args:
        client (pymongo.MongoClient): MongoDB client object.
        metadata (dict): Metadata for the model.

    Returns:
        None
    """
    # Access the desired database
    model_name = metadata['model_name']
    model_path = metadata['model_path']
    
    db = client[metadata['model_application']]

    # Create a new GridFS bucket
    fs = gridfs.GridFS(db, collection=metadata['model_format'])
    
    # Open the ONNX model file
    with open(model_path, 'rb') as f:

        # Calculate and store the checksum value
        model_data = f.read()
        checksum = calculate_checksum(model_data)
        metadata['checksum'] = checksum
        model_id = fs.put(model_data, filename=model_name, metadata=metadata)
    
    logger.info(f"Successfully stored the model with ID: {model_id}")

def save(query, model, path, meta_data):
        
    """
    Args:
        query (dict): query file
        model (binary): 

    Returns:
        
    """

    if query['metadata']['model_format']=='onnx':
        filename = query['metadata']['model_name'] + '.' + query['metadata']['model_format']
        logger.info(filename)
        filename_path = os.path.join(path,filename)
        # Download and save the file from GridFS
        with open(filename_path, 'wb') as f:
            f.write(model)
        filename_json = os.path.join(path, query['metadata']['model_name'] +'.json')
        with open(filename_json, "w") as outfile:
            json.dump(meta_data, outfile)
 

    if query['metadata']['model_format']=='pytorch':
        filename = query['metadata']['model_name'] + '.pt'
        logger.info(filename)
        filename_path = os.path.join(path,filename)
        # Download and save the file from GridFS
        with open(filename_path, 'wb') as f:
            f.write(model)

        filename_json = os.path.join(path, query['metadata']['model_name'] +'.json')
        with open(filename_json, "w") as outfile:
            json.dump(meta_data, outfile)

    if query['metadata']['model_format']=='artefact_zip':
        filename = query['metadata']['model_name'] + '.zip'

        logger.info(filename)
        filename_path = os.path.join(path,filename)
        # Download and save the file from GridFS
        with open(filename_path, 'wb') as f:
            f.write(model)
        filename_json = os.path.join(path, query['metadata']['model_name'] +'.json')
        with open(filename_json, "w") as outfile:
            json.dump(meta_data, outfile)

    if query['metadata']['model_format']=='onnx_zip':
        filename = query['metadata']['model_name'] + '.zip'

        logger.info(filename)
        filename_path = os.path.join(path,filename)
        # Download and save the file from GridFS
        with open(filename_path, 'wb') as f:
            f.write(model)
        filename_json = os.path.join(path, query['metadata']['model_name'] +'.json')
        with open(filename_json, "w") as outfile:
            json.dump(meta_data, outfile)


    if query['metadata']['model_format']=='pytorch_zip':
        filename = query['metadata']['model_name'] + '.zip'

        logger.info(filename)
        filename_path = os.path.join(path,filename)
        # Download and save the file from GridFS
        with open(filename_path, 'wb') as f:
            f.write(model)
        filename_json = os.path.join(path, query['metadata']['model_name'] +'.json')
        with open(filename_json, "w") as outfile:
            json.dump(meta_data, outfile)
 
    if query['metadata']['model_format']=='prototxt':
        filename = query['metadata']['model_name'] + '.prototxt'
        logger.info(filename)
        filename_path = os.path.join(path,filename)
        # Download and save the file from GridFS
        with open(filename_path, 'wb') as f:
            f.write(model)
        filename_json = os.path.join(path, query['metadata']['model_name'] +'.json')
        with open(filename_json, "w") as outfile:
            json.dump(meta_data, outfile)
 



def create_query(query_file, to_store):

    """
    Args:
        query_file (dict): query file

    Returns:
        query: query dict to in order to query the a given model

    """
    if to_store:

        query = {}
        for key, value in query_file.items():
        
            query[f"metadata.{key}"] = value
        
        
    else:

        model_application = query_file['model_application']
        model_name = query_file['model_name']
        architecture = query_file['architecture']
        model_format = query_file['model_format']
        version = query_file['version']
        loss = query_file['loss']
        quantization = query_file['quantization']
        training_data = query_file['training_data']

        query = {'metadata.model_application':model_application,
                'metadata.model_name': model_name,
                'metadata.architecture': architecture,
                'metadata.model_format': model_format,
                'metadata.version': version,
                'metadata.loss': loss,
                'metadata.quantization': quantization,
                'metadata.training_data': training_data,
                }

    return query


def model_search(client, query):
    """
    Args:
        query (dict): query to search for a model

    Returns:
        object_id: id of the model to query

    """

    model_application = query['metadata.model_application']
    collection = query['metadata.model_format']
    db = client[model_application]
    
    fs = gridfs.GridFS(db, collection= collection)
    
    # Query for specific documents
    collection = db[collection+".files"]
    result = collection.find_one(query)
    if result:
        logger.warning('Model is in the database')
        return fs, result
    else:
        logger.info("Model is not in the database")
        return None , None


def check_collections(metadata):

    """
    Check if the specified model application is valid.

    Args:
        metadata (dict): Metadata dictionary containing the model information.

    Returns:
        bool: True if the model application is valid, False otherwise.
    """

    model_application = metadata.get('model_application')
    if model_application not in dbconfig.required_collections:
        print("Invalid model application specified.\n")
        print("Required field are the following...\n", dbconfig.required_collections)
        return False
    
    return True

def get_username() -> str:
    return pwd.getpwuid(os.getuid())[0]


def generate_model_name(config:dict)-> dict:

    project_name = config["project_name"]
    model_application = config["model_application"]
    model_architecture = config["model_architecture"]
    model_version = config["model_version"]
    model_name = project_name + "_"+ model_application + "_" + model_architecture+ "_" + "V"+ str(model_version)
    config['model_name'] = model_name
    return config


def store(client:dict, metadata: dict)-> bool: 

    metadata['author'] = get_username()
    query = create_query(query_file=metadata, to_store=True)
    logger.info("Search for existing model...\n")
    # logger.info(f"client: {client}")
    # logger.info(f"query: {query}")
    _, result_local = model_search(client=client, query=query)
    if not result_local:
        logger.info("Model storing...\n")
        store_model(client=client, metadata=metadata)
        return True
    else:
        return False
    
def get(client, database, collection, model_id, save_path):

    db = client[database]
    
    fs = gridfs.GridFS(db, collection=collection)
    collection = collection+".files"
    collection = db[collection]

    model_object_id = ObjectId(model_id)
    result = collection.find_one({"_id": model_object_id})
    
    if result is not None:
        
        model_id = result['_id']
        logger.info("model_id",model_id)
        stored_checksum = result['metadata']['checksum']
        logger.info("stored_checksum",stored_checksum)
    
        model_data = fs.get(model_id).read()
        meta_data_info = result['metadata']
        
        calculated_checksum = calculate_checksum(model_data)
        logger.info("calculated_checksum",calculated_checksum)

        if stored_checksum == calculated_checksum:
            logger.info("Checksum verification succeeded...\n")
            save(query=result, model=model_data, path=save_path, meta_data = meta_data_info)
                
        else:
            logger.info("Checksum verification failed. Data might be corrupted.")
        return None
    else:
        logger.info("No model found!")