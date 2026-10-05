# Maize Nutrient Diagnosis API

FastAPI service serving a fine-tuned MobileNetV2 model (95.55% test accuracy)
that classifies a maize leaf photo into 6 nutrient-deficiency classes.

## Endpoint
POST /api/diagnose - multipart/form-data with a "file" field containing a JPG/PNG image.

## Confirmed working combo
Python 3.12.3 + tensorflow-cpu 2.21.0 (tested locally, loads and predicts correctly).
