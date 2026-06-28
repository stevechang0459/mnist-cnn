import dx_engine
import logging

logging.basicConfig(
    filename='dx_engine_api.log',
    level=logging.INFO,
    format='%(message)s',
    filemode='w'
)

def explore_dx_api():
    logging.info("Exploring dx_engine capabilities...\n")

    # 1. Print all public attributes in the main module
    logging.info("=== Module Attributes ===")
    for item in dir(dx_engine):
        if not item.startswith("__"):
            logging.info(f" - {item}")

    # 2. Inspect the InferenceEngine class if it exists
    if hasattr(dx_engine, 'InferenceEngine'):
        logging.info("\n=== InferenceEngine Methods ===")
        for item in dir(dx_engine.InferenceEngine):
            # Filter out standard Python magic methods for cleaner output
            if not item.startswith("__"):
                logging.info(f" - {item}")
    else:
        logging.info("\nInferenceEngine class not found in dx_engine.")

if __name__ == "__main__":
    explore_dx_api()
    print("API exploration complete. Please check 'dx_engine_api.log' for details.")
