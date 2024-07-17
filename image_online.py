import requests
def upload_image_to_server(image_path, username):
    url = "https://api.familystudy.cn:8090/file/upload"
    payload = {
        "username": "0001"
    }
    files = {
        'file': (image_path,
                 open(image_path, 'rb'), 'image/png'
                 )
    }
    response = requests.post(url, data=payload, files=files, verify=False)
    return response.text
# Example usage
image_path = 'uploads/20240201231130.png'
username = 'your_username'
uploaded_image_url = upload_image_to_server(image_path, username)
print("Uploaded Image URL:", uploaded_image_url)