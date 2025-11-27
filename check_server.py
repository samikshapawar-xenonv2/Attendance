import requests
import time

BASE_URL = "http://127.0.0.1:5000"

def check_server():
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✓ Server is running and accessible.")
        else:
            print(f"✗ Server returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server.")

if __name__ == "__main__":
    time.sleep(2) # Wait for server to start
    check_server()
