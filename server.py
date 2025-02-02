from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import base64
import os
from datetime import datetime

app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Folder to save images on the server
SAVE_FOLDER = "./saved_images"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

@app.get("/")
def home():
    return {"message": "Server is running"}

# WebSocket for real-time image transmission
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()  # Receive base64 image
            image_data = base64.b64decode(data)
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

             # Save the received image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(SAVE_FOLDER, f"image_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved image as {image_path}")

            # Display the received frame
            #cv2.imshow("Received Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        cv2.destroyAllWindows()
