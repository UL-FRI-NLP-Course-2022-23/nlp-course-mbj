import requests
import json
ROOT_URL = "http://www.dlib.si"

# Return response we get from URL
def get_response(URL):
    return requests.get(ROOT_URL + URL)

# Return response we get from URL and the page that was retrieved
def get_response_page(URL, page_counter):
    resp = get_response(URL+str(page_counter))
    current_page = int(resp.url.split("page=")[1])
    return [resp, current_page]

# Returns response we get from URL called with cookie
def get_response_cookie(URL, cookie):
    return requests.get(ROOT_URL + URL, allow_redirects=False, cookies=cookie)

# Saves array arr to file with name title
def save_json(title, arr):
    with open(title + ".json", 'w') as f:
        json.dump(arr, f, indent=2) 

# Reads array from file with name title
def load_json(title):
    with open(title + ".json", 'r') as f:
        res = json.load(f)
    return res

# Writes .txt file retrieved from the website to txt file
# First 3 lines are there because there are a lot of unecessery new lines and in most cases that should get rid of them
def write_txt(file_name, data, enc):
    data = data.replace("\r\n", "\n")
    data = data.replace("\n\n", "\n")
    data = data.replace("\n\n", "\n")
    with open(file_name, "w", encoding=enc) as f:
        f.write(data)
