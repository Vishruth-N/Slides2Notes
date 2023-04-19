import requests
from bs4 import BeautifulSoup

MOODLE_URL = "https://moodle.iitb.ac.in"
USERNAME = "200010088"
PASSWORD = "Vishruth@110402"

# Start a session to persist cookies
session = requests.Session()

# Log in to Moodle
login_url = f"{MOODLE_URL}/login/index.php"
login_data = {"username": USERNAME, "password": PASSWORD}
response = session.post(login_url, data=login_data)
COURSE_ID = "6671"
participants_url = f"{MOODLE_URL}/user/index.php?id={COURSE_ID}"
response = session.get(participants_url)
print(response.text)
soup = BeautifulSoup(response.text, "html.parser")
profile_urls = []

import re

def has_matching_id(tag):
    return tag.name == "th" and tag.has_attr("id") and re.match(r"user-index-participants-\d+", tag["id"])

# Look for the 'user' class attribute, adjust the class name based on your Moodle theme
user_elements = soup.find_all(has_matching_id)

for element in user_elements:
    print(element)
    exit(0)
    # Extract the profile URL from the 'href' attribute of the 'a' tag
    profile_url = element.find("a")["href"]
    profile_urls.append(profile_url)

print(profile_urls)

email_ids = []

for profile_url in profile_urls:
    response = session.get(profile_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Locate the 'mailto' link
    mailto_link = soup.find("a", href=lambda href: href and "mailto:" in href)

    if mailto_link:
        # Extract and decode the email address
        encoded_email = mailto_link["href"].replace("mailto:", "")
        email_id = bytes.fromhex(encoded_email).decode("utf-8")
        email_ids.append(email_id)

print(email_ids)

