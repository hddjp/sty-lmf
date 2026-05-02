import torch
import io
import argparse
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM
from fastapi.responses import Response
import logging
import time
import socket
import sys

from fastapi.responses import StreamingResponse

#if socket.socket().connect_ex(('127.0.0.1',8888))==0:
#    sys.exit()


sys.stdout = sys.stderr = open('/data/sty/sty-lmf/log1.txt', 'a', encoding='utf-8')
app = FastAPI()

try:
    s = socket.socket()
    s.bind(("127.0.0.1", 12345))
    print("bind success")
except:
    print("bind fail")
    sys.exit()



parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# @app.post("/infer")
# async def infer(request: Request):
#     data = await request.json()
#     t1=time.time()
#     inputs = {k: torch.tensor(v).to("cuda") for k, v in data.items()}
#     with torch.no_grad():
#         output = model(**inputs)
#     buffer = io.BytesIO()
#     torch.save(output.logits, buffer)
#     buffer.seek(0)
#     t2=time.time()
#     print(t2-t1)
#     return Response(content=buffer.read(), media_type="application/octet-stream")

@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    t1 = time.time()

    inputs = {k: torch.tensor(v).to("cuda") for k, v in data.items()}

    with torch.no_grad():
        output = model(**inputs)

    logits = output.logits.detach().to("cpu", non_blocking=True)

    buffer = io.BytesIO()
    torch.save(
        logits,
        buffer,
        _use_new_zipfile_serialization=False  # 🔥 关键优化
    )

    t2 = time.time()
    print("compute:", t2 - t1)

    return Response(
        content=buffer.getvalue(),   
        media_type="application/octet-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)