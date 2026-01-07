from fastapi import FastAPI

app = FastAPI()


@app.get("/hello/{user_input}")
async def read_hello(user_input: str):
    return {"message": f"Hello, World {user_input}"}

