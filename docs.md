# Docs

## Pre-requisites:
Make sure you have the following installed on your system:
- **Docker**
- **Python 3.x**
- **pip**

## Setup and Installation

Follow these steps to set up and run the project:

1. **Clone the Repository**
```bash
git clone git@github.com:mrospond/group_33.git
```
2. **Mount `group_33` Directory inside the `karmaresearch/wdps2` Docker Container**
```
cd group_33/
docker run -it -v ./:/home/user/workspace/ karmaresearch/wdps2
```
3. **Enter `workspace` dir**
```
cd workspace
pip3 install -r requirements.txt
```
4. **Setup Virtual Environment**
```
python3 -m venv venv # skip this step if already exists
source venv/bin/activate
```
5. **Install Dependencies**
```
pip3 install -r requirements.txt
```
6. **Run the Script**
```
python3 main.py
```

## Notes
- Ensure that Docker is running before executing Docker Command
- All dependencies required to run the project are listed in requirements.txt file
