"""Start the SAS API server. Usage: python -m sas.api.run"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("sas.api.main:app", host="0.0.0.0", port=8000, reload=True)
