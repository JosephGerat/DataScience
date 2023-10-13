
def download_spam_ds():
    from io import BytesIO
    import requests
    import tarfile

    BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
    FILES = ["20021010_easy_ham.tar.bz2",
             "20021010_hard_ham.tar.bz2",
             "20021010_spam.tar.bz2"]

    OUTPUT_DIR = 'spam_data'

    for filename in FILES:
        # Use requests to get the file contents at each URL.
        content = requests.get(f"{BASE_URL}/{filename}").content
        # Wrap the in-memory bytes so we can use them as a "file."
        fin = BytesIO(content)
        # And extract all the files to the specified output dir.
        with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
            tf.extractall(OUTPUT_DIR)

download_spam_ds()