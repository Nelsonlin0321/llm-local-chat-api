from datetime import datetime
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import baichuan2_13b_chat_4bits, rag


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PREFIX = "/api"

app.include_router(baichuan2_13b_chat_4bits.endpoints.router, prefix=PREFIX)
app.include_router(rag.endpoints.router, prefix=PREFIX)

app.mount("/static", StaticFiles(directory="./app/static"), name="static")

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
start_time = datetime.utcnow()
start_time = start_time.strftime(DATE_FORMAT)


@app.get(f"{PREFIX}/health-check")
async def health_check():
    response = f"The server is up since {start_time}"
    return {"message": response, "start_uct_time": start_time}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
