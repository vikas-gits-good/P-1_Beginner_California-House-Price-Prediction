from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion


def Main():
    try:
        logging.info("Initiating project")

        logging.info("Initiating Data Ingestion")
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()

        # logging.info("Initiating Data Transformation")
        # ...

        # logging.info("Initiating Model Training")
        # ...

        # logging.info("Best model saved and ready for user input prediction")

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    Main()
