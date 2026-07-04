from app.ml.pipelines.knee_oa_pipeline import KneeOAPipeline

class PredictionService:
    """
    Service to handle business logic for Knee Osteoarthritis predictions.
    Orchestrates the ML prediction pipeline and structures results.
    """
    def __init__(self):
        # The pipeline handles model registration, loading, and inference.
        self.pipeline = KneeOAPipeline()

    def predict_image(self, file_name: str, image_bytes: bytes) -> dict:
        """
        Receives raw image bytes, runs them through the pipeline, 
        and returns structured KL grade and class confidence results.
        """
        result = self.pipeline.predict(image_bytes)
        result["filename"] = file_name
        return result

# Singleton instance for the service layer
prediction_service = PredictionService()
