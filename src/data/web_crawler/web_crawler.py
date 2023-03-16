import os
import numpy as np
import time
from threading import Thread
from helper import save_json, load_json, write_txt, get_response, get_response_page, get_response_cookie

''''
All functions work with only subdirectories of URL's. Root part of URL (http://www.dlib.si) is defined in helper.py as ROOT_URL.
'''

# Function saves all of the records contained on all of the pages of a filtered query.
# URL - subdirectory of an URL you get after choosing category (e.g. books) and filters (e.g. language:slovenian, rights:public domain). 
# The last part of this subdirectory URL has to be 'page=' (without page number) so that we can iterate through them.
# Lastly you should set pageSize to 100
# E.g. /results/?query=%27rel%3dknjige%40OR%40rel%3dvisokošolska+dela%27&sortDir=ASC&sort=date&pageSize=100&flanguage=slv&faccess=public&page=
# With this URL we get all the records in public domain in slovenian language.
# file_name - file name under which we will save array of subdirectories of all the records we retrieved
def retrieve_books(URL, file_name):
    knjige_all = []
    unretrieved_pages = []
    page_counter = 1
    # Get respons of a page with URL and number of the page that was retrieved
    (resp, current_page) = get_response_page(URL, page_counter) 
    # If we try to go to a page that doesn't exist, we will be returned page number one, that's when we stop iteration
    while current_page == page_counter: 
        print(page_counter)
        if resp.status_code == 200:
            csvtext = resp.text.split('"')
            # Subdirectories contain /details in them, if it does we take it out and save it
            knjige = [i for i in csvtext if 'details' in i]
            # We remove every second occurence in table since each book has 2 occurrences of subdirectory link
            knjige = np.delete(knjige, list(range(0, len(knjige), 2)), axis=0)
            knjige_all.extend(knjige)
        else:
            unretrieved_pages.append(page_counter)
        page_counter = page_counter + 1
        (resp, current_page) = get_response_page(URL, page_counter)
    save_json(file_name, knjige_all)
    if len(unretrieved_pages) > 0:
        save_json("unretrieved_pages", unretrieved_pages)

# Function takes array of subdirectories, iterates over records and for each one it calls get file.
# It also takes directory into which it save books obtained from subdirectories
# This function produces next files:
# - successfully.json containing dictionary of value to title of all the records successfully retrieved, file 1.txt has title that corresponds to value in dict
# - unsuccessfully.json containing list of titles and reasons why we couldn't fetch the record:
#       Some of the reasons:
#       - duplicate: book with that title is already in knjige
#       - No txt file: records doesn't containt .txt file but something else (pdf, image...)
#       - error message: error occurred when retrieving record page,
#       - if we have URL and not title of the record, that means we couldn't retrieve page of the record, next to it is error message
def get_files(subdirectories, directory):
    successfully = {}
    unsuccessfully = {}
    counter = 1
    if not(os.path.exists(directory)):
        os.mkdir(directory)
    for URL in subdirectories:
        print(str(counter))
        counter = counter + 1
        resp = get_response(URL)
        if resp.status_code == 200:
            get_file(resp, successfully, unsuccessfully, directory)
        else:
            unsuccessfully[URL] = resp.status_code
        time.sleep(1)
        # Sleep for a second so there aren't to many requests.
    if len(successfully) > 0:
        save_json(directory + "\\successful", successfully)
    if len(unsuccessfully) > 0:
        save_json(directory + "\\unsuccessful", unsuccessfully)

# Funciton checks if record (response) contains .txt file
# If it does, it creates directory (which is the title of the record) in knjige
# It saves json file with metadata of record and txt file.
# If parsing the record fails, or if it doesn't contain .txt file we save records title and reason it failed to dictionary.
# This function produces next files:
# - directory(title of the record) in knjige containing .txt file (record) and .json containing metadata of the record
def get_file(resp, successfully, unsuccessfully, directory):
    # We need cookie, else the response doesn't return us text file but some html code.
    cookie = resp.cookies
    csvtext = resp.text.split('"')
    # We check if record is in public domain
    if 'javna domena' in csvtext:
        try:
            got_json = False
            got_txt = False
            for i in csvtext:
                if '<h1>' in i:
                    # Extract title of the record and remove break html tag from it
                    idx1 = i.find("h1>")
                    idx2 = i.find("</h1")
                    title = i[idx1+3:idx2]
                    title = title.replace("<br", "")
                    title = title.replace("/>", "")
                if 'JSON' in i :
                    # Save URL of JSON file
                    json_url = i
                    got_json = True
                if 'stream' in i and 'TEXT' in i:
                    # Save URl of .txt file
                    download_url = i
                    got_txt = True
            idx = str(len(successfully))
            path = directory + "\\" + idx
            # If we have .txt file save it else write this down.
            if got_txt and not(os.path.exists(path)):
                os.mkdir(path)
                path = path + "\\" + idx
                if got_json:
                    write_txt(path + ".json", get_response_cookie(json_url, cookie).text.replace(">", ">\n"))
                write_txt(path + ".txt", get_response_cookie(download_url, cookie).text)
                successfully[idx] = title
            elif not(got_txt):
                unsuccessfully[title] = "No txt file."
        except Exception as e:
            unsuccessfully[title] =e
    else:
        unsuccessfully[resp.url] = "Ni v javni domeni."

retrieve_books('/results/?query=%27rel%3dknjige%40OR%40rel%3dvisoko%C5%A1olska+dela%27&sortDir=ASC&sort=date&flanguage=slv&frights=PDM&pageSize=100&page=', 'seznam_knjig_slovenske_javno_dostopne')

# This runs get_files with n number of threads
def retrieve_threads(n):
    all_books = load_json('seznam_knjig_slovenske_javno_dostopne')
    num_books = len(all_books)
    for i in range(0,n):
        thread = (Thread(target=get_files, args=(all_books[i*num_books//n:(i+1)*num_books//n], "knjige" + str(i))))
        thread.start()
    
retrieve_threads(4)