# Copyright 2023 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#######################################################################################

# This example scripts runs openWakeWord in a simple web server receiving audio
# from a web page using websockets.

#######################################################################################

# Imports
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import FileResponse
import numpy as np
from openwakeword import Model
import resampy
import argparse
import json
import logging
import uvicorn
import asyncio

app = FastAPI()

default_wakeword_models = "./model/rnn/hi_aldelo_15000_adela.onnx"
default_inference_framework = "onnx"


def create_model_instance():
    """Return a new Model instance for a websocket connection.

    Uses `default_wakeword_models` and `default_inference_framework` if set.
    """
    if default_wakeword_models:
        return Model(
            wakeword_models=[default_wakeword_models],
            inference_framework=default_inference_framework,
            # vad_threshold=0.5,
        )
    return Model(inference_framework=default_inference_framework)


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    print("New WebSocket connection")
    await websocket.accept()

    # Create a per-connection model instance and send loaded models to the client
    model = None
    try:
        model = create_model_instance()
        await websocket.send_text(
            json.dumps({"loaded_models": list(model.models.keys())})
        )
        print(f"Loaded models (connection): {list(model.models.keys())}")
    except Exception as e:
        print(f"Failed to create/send loaded models for connection: {e}")
        # If we couldn't create a model for this connection, close socket
        try:
            await websocket.close()
        except Exception:
            pass
        return

    sample_rate = 16000
    try:
        while True:
            msg = await websocket.receive()
            # msg is a dict like {'type': 'websocket.receive', 'bytes'| 'text': ...}
            if "text" in msg and msg["text"] is not None:
                # Expecting sample rate as text
                try:
                    sample_rate = int(msg["text"])
                except Exception:
                    print(f"Received text message: {msg['text']}")
                continue

            if "bytes" in msg and msg["bytes"] is not None:
                audio_bytes = msg["bytes"]
                if len(audio_bytes) % 2 == 1:
                    audio_bytes += b"\x00"

                data = np.frombuffer(audio_bytes, dtype=np.int16)
                if sample_rate != 16000:
                    data = resampy.resample(data, sample_rate, 16000)
                # print(f"Received {data.size} samples")
                # Offload the potentially blocking predict call to a threadpool
                # to avoid blocking the event loop. Each connection uses its
                # own model instance so no additional serialization is needed.
                loop = asyncio.get_event_loop()
                predictions = await loop.run_in_executor(None, model.predict, data)

                activations = []
                for key in predictions:
                    if predictions[key] >= 0.5:
                        activations.append(key)

                if activations:
                    print(f"Detected: {activations}")
                    try:
                        await websocket.send_text(
                            json.dumps({"activations": activations})
                        )
                    except Exception as e:
                        print(f"Failed to send activations: {e}")
            else:
                # Other message types (close, etc.) will break the loop
                if msg.get("type") == "websocket.disconnect":
                    break
    except Exception as exc:
        print(f"WebSocket error: {exc}")
    finally:
        # Try to clean up the per-connection model if it exposes a close()
        if model is not None:
            close_fn = getattr(model, "close", None)
            try:
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass


@app.get("/")
async def index(request: Request):
    return FileResponse("index.html")


def run_uvicorn(host: str, port: int):
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        help="Host to bind the server",
        type=str,
        default="0.0.0.0",
        required=False,
    )
    parser.add_argument(
        "--port",
        help="Port to bind the server",
        type=int,
        default=9000,
        required=False,
    )
    args = parser.parse_args()

    # Run uvicorn
    run_uvicorn(args.host, args.port)
