# Docs

### How to run:

1. Clone the repository
```
git clone git@github.com:mrospond/group_33.git
```

2. Mount `group_33` directory inside the `karmaresearch/wdps2` docker container
```
docker run -it -v ./group_33/:/home/user/workspace/ karmaresearch/wdps2
```

3. Enter `workspace` dir and setup virtual environment
```
cd workspace
source venv/bin/activate
pip3 install -r requirements.txt
```

4. Run the script
```
python3 main.py
```
