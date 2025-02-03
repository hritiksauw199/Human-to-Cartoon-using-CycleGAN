const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const ws = new WebSocket("wss://promoted-fit-mako.ngrok-free.app/ws"); // Replace with your ngrok URL

// Request camera access
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error("Error accessing camera:", error);
        alert("Please allow camera access in your browser settings.");
    }
}

// Start camera when the page loads
document.addEventListener("DOMContentLoaded", startCamera);

// Capture frame and send to server
setInterval(() => {
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        console.log("Sending image to server", imageData); // Log image data
        const imageData = canvas.toDataURL("image/jpeg").split(",")[1];
        ws.send(imageData);
    }
}, 100);

// Send captured image to server
function sendImageToServer() {
    
    const imageData = canvas.toDataURL("image/jpeg").split(",")[1];
    console.log("Sending image to server", imageData); // Log image data
    ws.send(imageData);  // Send image to WebSocket server
}

// Button to save the image to the server
const saveButton = document.createElement('button');
saveButton.innerText = 'Send Image to Server';
saveButton.onclick = sendImageToServer;
document.body.appendChild(saveButton);
