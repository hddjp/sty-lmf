import torch
import io
import argparse
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM
from fastapi.responses import Response
import logging


import sys
sys.stdout = sys.stderr = open('/data/sty/sty-lmf/log.txt', 'a', encoding='utf-8')


app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    inputs = {k: torch.tensor(v).to("cuda") for k, v in data.items()}
    with torch.no_grad():
        output = model(**inputs)
    buffer = io.BytesIO()
    torch.save(output.logits, buffer)
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8888)